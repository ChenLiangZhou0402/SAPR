"""List class indices or check the crop neighbors against a training manifest."""

import argparse
import csv
from pathlib import Path

import numpy as np


DATASETS = ("coco", "nus", "genome")
PRESETS = Path(__file__).resolve().parent / "presets"


def manifest(path, classes, nonempty=False):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"image", "labels"}.issubset(reader.fieldnames or ()):
            raise ValueError("Expected the CSV header image,labels")
        for row in reader:
            if not row["image"] or row["labels"] is None:
                raise ValueError(f"Incomplete image row on CSV line {reader.line_num}")
            indices = [int(value) for value in row["labels"].split()]
            if any(index < 0 or index >= classes for index in indices):
                raise ValueError(f"Class index outside the preset on CSV line {reader.line_num}")
            if nonempty and not indices:
                continue
            truth = np.zeros(classes, dtype=np.uint8)
            truth[indices] = 1
            yield row["image"], truth


def neighbors(path, classes):
    rows = [truth for _, truth in manifest(path, classes)]
    count = len(rows) // 100
    if count == 0:
        raise ValueError("A one-percent subset needs at least 100 training images")
    sample = np.random.default_rng(1).permutation(len(rows))[:count]
    labels = np.stack([rows[i] for i in sample]).astype(np.int64)
    joint = labels.T @ labels
    np.fill_diagonal(joint, 0)
    # For each target the denominator is fixed, so joint counts rank P(neighbor | target).
    order = np.argsort(-joint, axis=1, kind="stable")[:, :3]
    selected = np.where(np.take_along_axis(joint, order, axis=1) >= 3, order, -1)
    return selected, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="coco")
    parser.add_argument("--train", type=Path, help="Check crop neighbors using a full training CSV")
    args = parser.parse_args()
    with np.load(PRESETS / (args.dataset + ".npz"), allow_pickle=False) as preset:
        labels = preset["labels"].tolist()
        expected = preset["neighbors"]
    if args.train:
        result, count = neighbors(args.train, len(labels))
        different = np.flatnonzero(np.any(result != expected, axis=1))
        if len(different):
            names = ", ".join(labels[i] for i in different)
            parser.exit(1, f"Crop neighbors differ for {names}\n")
        print(f"{count} training images sampled. Crop neighbors match for all {len(labels)} classes.")
    else:
        for index, name in enumerate(labels):
            print(f"{index}\t{name}")


if __name__ == "__main__":
    main()
