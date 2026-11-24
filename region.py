"""Select class-specific regions and score them on the original pixels."""

from math import ceil, floor

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage
import torch
from torch.nn import functional as F

from vision import autocast


FULL = (0, 0, 224, 224, 0, 0, 0, 0)
FILL = (123, 117, 104)


def extract(model, image):
    """Use standard CLS and a self-attending last block for local patch features."""
    visual = model.visual
    patches = visual.conv1(image.to(visual.conv1.weight.dtype)).flatten(2).transpose(1, 2)
    token = visual.class_embedding.to(patches.dtype).expand(patches.size(0), 1, -1)
    sequence = torch.cat((token, patches), dim=1) + visual.positional_embedding.to(patches.dtype)
    sequence = visual.ln_pre(sequence).transpose(0, 1)
    for layer in visual.transformer.resblocks[:-1]:
        sequence = layer(sequence)

    last = visual.transformer.resblocks[-1]
    whole = last(sequence)
    query = last.ln_1(sequence)
    mask = query.new_full((len(query), len(query)), -100).fill_diagonal_(0)
    patch = sequence + last.attn(query, query, query, attn_mask=mask, need_weights=False)[0]
    patch = patch + last.mlp(last.ln_2(patch))

    whole = visual.ln_post(whole.transpose(0, 1)) @ visual.proj
    patch = visual.ln_post(patch.transpose(0, 1)) @ visual.proj
    return F.normalize(whole.float(), dim=-1)[:, 0], F.normalize(patch.float(), dim=-1)[:, 1:]


def select(maps, related):
    target, distinct, context = maps
    seed = (target > .5) & (distinct > .5)
    support = ((context > .5) | (target > .5)) if related else (target > .5)
    labels, count = ndimage.label(support)
    if not count:
        return None
    # The frozen selector breaks equal masses by area, then raster order.
    sizes = np.bincount(labels.ravel())[1:]
    order = np.argsort(-sizes, kind="stable") + 1
    if not related:
        return labels == order[0]
    best, best_mass = None, 0.0
    for component in order:
        inside = labels == component
        mass = (target[inside] * distinct[inside] * seed[inside]).sum()
        if mass > best_mass:
            best, best_mass = inside, mass
    return best


def geometry(component):
    y, x = np.nonzero(component)
    x0, x1 = 16 * int(x.min()), 16 * (int(x.max()) + 1)
    y0, y1 = 16 * int(y.min()), 16 * (int(y.max()) + 1)
    dx, dy = .1 * (x1 - x0), .1 * (y1 - y0)
    x0, x1 = max(0, floor(x0 - dx)), min(224, ceil(x1 + dx))
    y0, y1 = max(0, floor(y0 - dy)), min(224, ceil(y1 + dy))
    width, height = x1 - x0, y1 - y0
    side = max(width, height)
    left, top = (side - width) // 2, (side - height) // 2
    return x0, y0, x1, y1, left, top, side - width - left, side - height - top


def render(image, box):
    width, height = image.size
    rw, rh = (int(224 * width / height), 224) if height < width else (224, int(224 * height / width))
    left, top = round((rw - 224) / 2), round((rh - 224) / 2)
    x0, y0, x1, y1, *padding = box
    frame = (max(0, floor((left + x0) * width / rw)), max(0, floor((top + y0) * height / rh)),
             min(width, ceil((left + x1) * width / rw)), min(height, ceil((top + y1) * height / rh)))
    border = tuple(round(n * ratio) for n, ratio in zip(padding, (width/rw, height/rh) * 2))
    region = image.crop(frame)
    if any(border):
        region = ImageOps.expand(region, border, fill=FILL)
    return region.resize((224, 224), Image.Resampling.BICUBIC)


def verify(image, model, preprocess, templates, neighbors, device):
    with autocast(device):
        whole, patches = extract(model, preprocess(image).unsqueeze(0).to(device))
    global_scores = (100 * whole @ templates[0].T).cpu().softmax(-1).numpy()[0]
    logits = 100 * torch.einsum("bpd,fcd->bfcp", patches, templates)
    probability = logits.softmax(2).reshape(3, -1, 14, 14)
    lower = probability.amin((-2, -1), keepdim=True)
    span = probability.amax((-2, -1), keepdim=True) - lower
    fields = torch.where(span > 1e-12, (probability-lower) / span.clamp_min(1e-12), 0).cpu().numpy()
    ordinary = probability[0].cpu().numpy()
    threshold = ordinary.min() + .5 * (ordinary.max() - ordinary.min())
    candidates = np.flatnonzero(ordinary.max(axis=(1, 2)) > threshold)
    local = np.zeros_like(global_scores)
    regions = {}
    for target in candidates:
        component = select(fields[:, target], bool(len(neighbors[target])))
        if component is None:
            continue
        box = geometry(component)
        if box == FULL:
            local[target] = global_scores[target]
        else:
            regions.setdefault(box, []).append(target)

    if regions:
        crops = torch.stack([preprocess(render(image, box)) for box in regions]).to(device)
        with autocast(device):
            features = model.encode_image(crops)
        features = F.normalize(features.float(), dim=-1)
        probabilities = (100 * features @ templates[0].T).cpu().softmax(-1).numpy()
        for scores, targets in zip(probabilities, regions.values()):
            for target in targets:
                adjacent = neighbors[target]
                active = adjacent[global_scores[adjacent] > 1 / len(global_scores)]
                excluded = scores[active].sum(dtype=np.float64)
                value = float(scores[target])
                denominator = max(1 - excluded, value)
                local[target] = value / denominator if denominator > 0 else 0.
    return global_scores, local
