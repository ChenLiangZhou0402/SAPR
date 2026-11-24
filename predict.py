"""Classify an image with a frozen SAPR preset."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from labels import DATASETS
from region import verify
from score import fuse, prompt
from tag import identify, locate
from vision import encode, load


def options(parser):
    parser.add_argument("--dataset", choices=DATASETS, default="coco")
    parser.add_argument("--backend", type=Path, default=Path(__file__).parent / "vendor",
                        help="Path to the TagCLIP checkout")
    parser.add_argument("--checkpoint", type=Path, help="Path to the OpenAI ViT-B-16.pt checkpoint")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "sapr",
                        help="Directory for reusable text embeddings")


@torch.inference_mode()
def predict(image, encoder):
    native, field = locate(image, encoder)
    rows, cols = native.shape[-2] // 16, native.shape[-1] // 16
    classes = len(encoder.labels)
    maps = field[:, :classes].float().cpu().numpy().reshape(rows, cols, classes).clip(0)
    prompts = prompt(encode(image, encoder), encoder.prompts, maps, image.size)
    tags = identify(native, field, encoder)
    global_scores, local = verify(image, encoder.model, encoder.preprocess,
                                  encoder.templates, encoder.neighbors, encoder.device)
    return fuse(global_scores, prompts, tags, local, encoder.omega)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--top", type=int, default=5, help="Number of ranked labels to print")
    options(parser)
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    encoder = load(args.dataset, args.backend, args.checkpoint, args.device, args.cache)
    with Image.open(args.image) as source:
        scores = predict(source.convert("RGB"), encoder)
    order = np.argsort(-scores, kind="stable")[:args.top]
    print(json.dumps([{"label": encoder.labels[i], "score": float(scores[i])} for i in order], indent=2))


if __name__ == "__main__":
    main()
