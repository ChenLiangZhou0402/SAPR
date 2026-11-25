# Usage

## Install

Use Python 3.10–3.12 and Git. For GPU inference, install matching CUDA builds of PyTorch and torchvision first.

```sh
git clone https://github.com/ChenLiangZhou0402/SAPR.git
cd SAPR
python -m pip install -r requirements.txt
git clone https://github.com/linyq2117/TagCLIP.git vendor
git -C vendor checkout c32b640f3a1cf840e23e4ca8b55aa2b35d086fc5
```

## Predict

```sh
python predict.py image.jpg --dataset coco --top 5
```

Choose `coco`, `nus`, or `genome`. The matching preset is included in `presets/`. The command prints labels and ranking scores in descending order. The first run downloads the OpenAI CLIP ViT-B/16 checkpoint and caches TagCLIP text embeddings for the selected dataset. To use a local checkpoint on CPU, run

```sh
python predict.py image.jpg --dataset coco --checkpoint ViT-B-16.pt --device cpu
```

## Evaluate

Each dataset has two annotation files under `data/<dataset>/`. Use `test.csv` for evaluation and `train.csv` for the fixed 1% training subset. Supply the directory containing your images via `--root`.

| Dataset | Image paths inside the evaluation root | Training subset | Evaluation images |
|---|---|---:|---:|
| COCO (`coco`) | `COCO_val2014_*.jpg` | 820 | 40,137 |
| NUS-WIDE (`nus`) | `images/<folder>/<filename>.jpg` | 1,617 | 83,898 |
| VG-256 (`genome`) | `images/<id>.jpg` | 746 | 32,011 |

```sh
python evaluate.py data/coco/test.csv --root path/to/val2014 --dataset coco --save scores.npz
```

Evaluation reports mAP over classes with both positive and negative images. The NUS-WIDE evaluation file contains images with at least one target label. Omit `--save` to print the result only.

For your own images, use a UTF-8 CSV with relative image paths and space-separated, zero-based label indices. An empty label field represents an image with no target labels.

```csv
image,labels
images/first.jpg,0 2
images/second.jpg,1
images/third.jpg,
```

To print the class order for a preset, run

```sh
python labels.py --dataset coco
```

To verify the crop neighbors from the included training annotations, run

```sh
python labels.py --dataset coco --subset data/coco/train.csv
```

If supplying a complete training split in its original row order, use `--train full.csv` to recreate the 1% sample.
