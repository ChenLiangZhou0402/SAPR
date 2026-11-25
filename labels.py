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


def neighbors(path, classes, *, subset=False):
    rows = [truth for _, truth in manifest(path, classes)]
    if not subset:
        count = len(rows) // 100
        if count == 0:
            raise ValueError("A one-percent subset needs at least 100 training images")
        sample = np.random.default_rng(1).permutation(len(rows))[:count]
        rows = [rows[i] for i in sample]
    if not rows:
        raise ValueError("The selected training subset is empty")
    labels = np.stack(rows).astype(np.int64)
    joint = labels.T @ labels
    np.fill_diagonal(joint, 0)
    # For each target the denominator is fixed, so joint counts rank P(neighbor | target).
    order = np.argsort(-joint, axis=1, kind="stable")[:, :3]
    selected = np.where(np.take_along_axis(joint, order, axis=1) >= 3, order, -1)
    return selected, len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="coco")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train", type=Path, help="Draw the fixed one-percent subset from a full training CSV")
    group.add_argument("--subset", type=Path, help="Use all rows in an already selected training subset")
    args = parser.parse_args()
    with np.load(PRESETS / (args.dataset + ".npz"), allow_pickle=False) as preset:
        labels = preset["labels"].tolist()
        expected = preset["neighbors"]
    source = args.subset or args.train
    if source:
        result, count = neighbors(source, len(labels), subset=args.subset is not None)
        different = np.flatnonzero(np.any(result != expected, axis=1))
        if len(different):
            names = ", ".join(labels[i] for i in different)
            parser.exit(1, f"Crop neighbors differ for {names}\n")
        print(f"Crop neighbors match for all {len(labels)} classes using {count} training images.")
    else:
        for index, name in enumerate(labels):
            print(f"{index}\t{name}")


if __name__ == "__main__":
    main()
