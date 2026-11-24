# Usage

## Install

From the repository directory, use Python 3.10–3.12 and Git. For GPU inference, install matching CUDA builds of PyTorch and torchvision first.

```sh
python -m pip install -r requirements.txt
git clone https://github.com/linyq2117/TagCLIP.git vendor
git -C vendor checkout c32b640f3a1cf840e23e4ca8b55aa2b35d086fc5
```

## Predict

Place the preset for your dataset in the `presets/` folder at the repository root. Use `coco.npz`, `nus.npz`, or `genome.npz` from your SAPR code package.

```sh
python predict.py image.jpg --dataset coco --top 5
```

Choose `coco` for COCO, `nus` for NUS-WIDE, or `genome` for VG-256. The command prints labels and ranking scores in descending order. The first run downloads the OpenAI CLIP ViT-B/16 checkpoint and caches the dataset’s text embeddings. To use a local checkpoint or run on CPU:

```sh
python predict.py image.jpg --dataset coco --checkpoint ViT-B-16.pt --device cpu
```

## Evaluate

Create a UTF-8 CSV with one image per row. List its zero-based class indices separated by spaces. An empty label field represents an image with no target labels.

```csv
image,labels
images/first.jpg,0 2
images/second.jpg,1
images/third.jpg,
```

To print the class order for your preset:

```sh
python labels.py --dataset coco
```

Run evaluation with paths relative to `--root`:

```sh
python evaluate.py labels.csv --root data --dataset coco --save scores.npz
```

The command reports mAP over classes with both positive and negative images. The NUS-WIDE preset evaluates images with at least one target label. Omit `--save` to print the result only.

To verify the crop-neighbor table, supply the full training split in its original row order using the same CSV format:

```sh
python labels.py --dataset coco --train train.csv
```
