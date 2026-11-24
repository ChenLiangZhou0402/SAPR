"""Load frozen encoders and render the registered prompt windows."""

from contextlib import nullcontext
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
from pathlib import Path
from tempfile import NamedTemporaryFile

import clip
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from labels import PRESETS
from score import bounds, scale, unit


CHECKSUM = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"


@dataclass
class Encoder:
    model: torch.nn.Module
    regional: torch.nn.Module
    preprocess: object
    templates: torch.Tensor
    prompts: tuple
    text: torch.Tensor
    labels: list
    neighbors: list
    omega: float
    device: torch.device


def autocast(device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def backend(folder):
    # TagCLIP uses an absolute import from clip.multi_head_attn inside its model.
    root = (folder / "clip").resolve()
    if not (root / "build_model.py").is_file():
        raise FileNotFoundError(f"Expected the TagCLIP checkout in {folder}. See README.md.")
    if str(root) not in clip.__path__:
        clip.__path__.append(str(root))
    spec = importlib.util.spec_from_file_location("clip.regional", root / "clip.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def vocabulary(module, model, names, device, cache):
    # CPU and CUDA use different precision paths. Keep their text caches separate.
    identity = "\n".join(["ViT-B/16", "c32b640", str(device.type), torch.__version__, *names])
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (sha256(identity.encode()).hexdigest() + ".npy")
    if path.is_file():
        values = np.load(path, allow_pickle=False)
        if values.shape != (len(names), 512) or not np.isfinite(values).all():
            raise ValueError(f"Invalid text cache at {path}")
        return torch.from_numpy(values).to(device)
    values = F.normalize(module.encode_text_with_prompt_ensemble(model, names, device), dim=-1)
    with NamedTemporaryFile(dir=cache, suffix=".npy", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.save(stream, values.detach().cpu().numpy(), allow_pickle=False)
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return values


@torch.inference_mode()
def load(dataset, folder, checkpoint, device, cache):
    device = torch.device(device)
    module = backend(Path(folder))
    if checkpoint:
        checksum = sha256()
        with Path(checkpoint).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                checksum.update(chunk)
        if checksum.hexdigest() != CHECKSUM:
            raise ValueError("The supplied checkpoint must be the original OpenAI CLIP ViT-B/16.")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.set_num_threads(min(torch.get_num_threads(), 8))
    model, preprocess = clip.load(str(checkpoint) if checkpoint else "ViT-B/16", device=device, jit=False)
    if model.visual.conv1.kernel_size != (16, 16) or len(model.visual.transformer.resblocks) != 12:
        raise ValueError("The preset requires the OpenAI CLIP ViT-B/16 checkpoint.")
    path = Path(checkpoint) if checkpoint else Path.home() / ".cache/clip/ViT-B-16.pt"
    regional, _ = module.load(str(path), device=device, jit=False)
    model.eval().requires_grad_(False)
    regional.eval().requires_grad_(False)
    with np.load(PRESETS / (dataset + ".npz"), allow_pickle=False) as preset:
        labels = preset["labels"].tolist()
        names = preset["names"].tolist()
        neighbors = [row[row >= 0] for row in preset["neighbors"]]
        templates = torch.from_numpy(preset["templates"]).to(device)
        prompts = (unit(preset["distinct"]), unit(preset["context"]))
        omega = float(preset["weight"])
    text = vocabulary(module, regional, names, device, Path(cache))
    return Encoder(model, regional, preprocess, templates, prompts, text, labels, neighbors, omega, device)


def windows(image):
    # The registered grid spans 222 of 224 reference pixels along each axis.
    full = image.resize((224, 224), Image.Resampling.BICUBIC)
    source = image if scale(image.size) > 0 else full
    views = [full]
    for left, right in bounds(source.width):
        for top, bottom in bounds(source.height):
            box = left, top, max(left + 1, right), max(top + 1, bottom)
            views.append(source.crop(box).resize((224, 224), Image.Resampling.BICUBIC))
    return views


def encode(image, encoder):
    batch = torch.stack([encoder.preprocess(view) for view in windows(image)]).to(encoder.device)
    return encoder.model.encode_image(batch).float().cpu().numpy()
