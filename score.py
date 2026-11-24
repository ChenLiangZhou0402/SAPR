"""Pool prompt windows and combine the four class scores."""

import numpy as np


def unit(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True).clip(1e-12)


def softmax(values, axis):
    exponentials = np.exp(values - values.max(axis=axis, keepdims=True))
    return exponentials / exponentials.sum(axis=axis, keepdims=True)


def bounds(length):
    edges = [round(length * 74 * i / 224) for i in range(4)]
    return list(zip(edges, edges[1:]))


def overlap(length, patches):
    """Length of each window's intersection with each native patch."""
    edges = np.arange(patches + 1, dtype=np.float64) * length / patches
    return np.stack([
        np.maximum(0, np.minimum(stop, edges[1:]) - np.maximum(start, edges[:-1]))
        for start, stop in bounds(length)
    ]).astype(np.float32)


def weights(field, size):
    height, width, classes = field.shape
    excess = np.maximum(field - field.mean(axis=(0, 1), keepdims=True), 0)
    vertical = overlap(size[1], height) @ excess.reshape(height, -1)
    mass = np.einsum("xc,yck->xyk", overlap(size[0], width), vertical.reshape(3, width, classes))
    mass = mass.reshape(9, classes) / np.float32(size[0] * size[1])
    total = mass.sum(axis=0, keepdims=True)
    return np.divide(mass, total, out=np.full_like(mass, 1 / 9), where=total > 0)


def scale(size):
    width, height = size
    patches = ((width + 15) // 16) * ((height + 15) // 16)
    return np.float32(max(0, 1 - 196 / patches))


def prompt(features, banks, field, size):
    """Read one full view and nine windows in column-major spatial order."""
    features = unit(features)
    discriminative = features @ banks[0].T
    correlative = features @ banks[1].T
    strength = scale(size)
    if strength == 0:
        # Preserve the original two-bank pooling order on inactive images.
        first = discriminative[0] + discriminative[1:].mean(axis=0)
        second = correlative[0] + correlative[1:].mean(axis=0)
        return softmax(np.float32(100) * (np.float32(.25) * first + np.float32(.25) * second), 0)

    logits = np.float32(50) * (discriminative + correlative)
    uniform = softmax(.5 * logits[0] + .5 * ((1 / 9) * logits[1:]).sum(axis=0), 0)
    probability = softmax(logits, 1)
    aligned = .5 * probability[0] + .5 * (weights(field, size) * probability[1:]).sum(axis=0)
    return uniform + strength * (aligned - uniform)


def fuse(global_scores, prompts, tags, local, omega):
    """Preserve the floating-point accumulation order of the frozen predictor."""
    anchor = global_scores + np.float32(omega) * tags
    mixed = anchor.astype(np.float64) + .8 * (prompts.astype(np.float64) - global_scores)
    return mixed.astype(np.float32) + np.float32(.25) * local
