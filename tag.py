"""Read the native TagCLIP class map and verify its candidate regions."""

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as transforms


def refine(coarse, attention, height, width):
    initial = attention @ coarse
    low, high = initial.amin(0), initial.amax(0)
    span = high - low
    normal = (initial - low) / span.clamp_min(torch.finfo(initial.dtype).tiny)
    maps = normal.cpu().numpy().reshape(height, width, -1)
    cutoffs = normal.mean(0).cpu().numpy()
    support = np.zeros_like(maps, dtype=np.uint8)
    for category in np.flatnonzero((high > 0).cpu().numpy()):
        pixels = (255 * maps[:, :, category]).astype(np.uint8)
        cutoff = int(cutoffs[category] * pixels.max())
        mask = cv2.threshold(pixels, cutoff, 255, cv2.THRESH_BINARY)[1]
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Preserve the author's exclusive crop and last-index clamp.
            support[y:min(y+h, height-1), x:min(x+w, width-1), category] = 1
    support = torch.from_numpy(support.reshape(initial.shape)).to(coarse)
    return torch.where(high > 0, attention @ (coarse * support), initial)


def locate(image, encoder):
    width, height = image.size
    rows, cols = (height + 15) // 16, (width + 15) // 16
    resized = image.resize((16 * cols, 16 * rows), Image.Resampling.BICUBIC)
    native = transforms.normalize(transforms.to_tensor(resized),
                                  (.48145466, .4578275, .40821073),
                                  (.26862954, .26130258, .27577711)).unsqueeze(0).to(encoder.device)
    model = encoder.regional
    tokens, layers = model.encode_image_tagclip(native, 16*rows, 16*cols, attn_mask=1)
    tokens = tokens / tokens.norm(dim=-1, keepdim=True)
    layers = torch.stack([layer[:, 1:, 1:] for layer in layers])[:, 0]
    cutoffs = layers.reshape(len(layers), -1).mean(-1)[:, None, None]
    attention = layers[8:-1].mean(0) * ((layers > cutoffs).sum(0) >= 4).float()
    coarse = (model.logit_scale.exp() * tokens @ encoder.text.T)[0, 1:].softmax(-1)
    field = refine(coarse, attention, rows, cols)
    return native, field


def identify(native, field, encoder):
    model = encoder.regional
    height, width = native.shape[-2] // 16, native.shape[-1] // 16
    maxima = field.amax(0)[:len(encoder.labels)].clone()
    threshold = field.min() + .5 * (field.max() - field.min())
    for category in torch.where(maxima > threshold)[0].tolist():
        values = field[:, category].reshape(height, width)
        mask = (values - values.min()) > .5 * (values.max() - values.min())
        ys, xs = torch.where(mask)
        if len(xs) == 0:
            break
        left, right = int(xs.min()), min(int(xs.max()) + 1, width - 1)
        top, bottom = int(ys.min()), min(int(ys.max()) + 1, height - 1)
        if right == left or bottom == top:
            break
        crop = native[:, :, 16*top:16*bottom, 16*left:16*right]
        crop = F.interpolate(crop, (224, 224), mode="bilinear", align_corners=False)
        support = mask[top:bottom, left:right].float()[None, None]
        keep = F.interpolate(support, (14, 14), mode="nearest").flatten().bool()
        attention = native.new_zeros((197, 197))
        attention[0, 1:] = (~keep).to(native.dtype) * -1000
        tokens = model.encode_image_tagclip(crop, 224, 224, attn_mask=attention)[0]
        tokens = tokens / tokens.norm(dim=-1, keepdim=True)
        scores = (model.logit_scale.exp() * tokens @ encoder.text.T)[:, 0].softmax(-1)
        maxima[category] = .5 * maxima[category] + .5 * scores[0, category]
    return maxima.float().cpu().numpy()
