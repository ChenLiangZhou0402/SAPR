"""Evaluate multi-label recognition on a CSV image manifest."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score

from labels import manifest
from predict import options, predict
from vision import load


def average(labels, scores):
    eligible = labels.any(axis=0) & ~labels.all(axis=0)
    if not eligible.any():
        raise ValueError("Evaluation needs at least one class with both positive and negative images")
    return 100 * average_precision_score(labels[:, eligible], scores[:, eligible], average="macro")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=Path("."), help="Root of the image paths in the CSV")
    parser.add_argument("--save", type=Path, help="Optional output .npz for scores, labels and image names")
    options(parser)
    args = parser.parse_args()
    encoder = load(args.dataset, args.backend, args.checkpoint, args.device, args.cache)
    names, labels, scores = [], [], []
    for name, truth in manifest(args.manifest, len(encoder.labels), args.dataset == "nus"):
        with Image.open(args.root / name) as source:
            scores.append(predict(source.convert("RGB"), encoder))
        labels.append(truth)
        names.append(name)
    if not names:
        parser.error("No eligible images in the manifest")
    labels, scores = np.stack(labels), np.stack(scores)
    result = average(labels, scores)
    if args.save:
        with args.save.open("wb") as output:
            np.savez_compressed(output, scores=scores, labels=labels, images=names, classes=encoder.labels)
    print(f"{len(names)} images, mAP {result:.3f}%")


if __name__ == "__main__":
    main()
