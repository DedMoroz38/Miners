"""Full-image / tiled inference producing a phase class map + uncertainty."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from .dataset import IMAGENET_MEAN, IMAGENET_STD
from .model import forward_logits
from .preprocess import PreprocessConfig, preprocess


def _to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)


def _tiles(h: int, w: int, tile: int, overlap: float):
    step = int(tile * (1 - overlap))
    ys = list(range(0, max(1, h - tile + 1), step)) or [0]
    xs = list(range(0, max(1, w - tile + 1), step)) or [0]
    if ys[-1] != h - tile:
        ys.append(max(0, h - tile))
    if xs[-1] != w - tile:
        xs.append(max(0, w - tile))
    for y in ys:
        for x in xs:
            yield y, x


def _ramp(tile: int) -> np.ndarray:
    r = np.hanning(tile)
    return np.clip(np.outer(r, r), 1e-3, None).astype(np.float32)


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    image_bgr: np.ndarray,
    device: torch.device,
    n_classes: int,
    tile: int = 1024,
    overlap: float = 0.25,
    pre_cfg: PreprocessConfig = PreprocessConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Return (class_map HxW uint8, uncertainty HxW float in [0,1])."""
    model.eval()
    img = preprocess(image_bgr, pre_cfg)
    h, w = img.shape[:2]

    if max(h, w) <= tile:
        logits = forward_logits(model, _to_tensor(img).to(device))[0]
        prob = torch.softmax(logits, 0).cpu().numpy()
    else:
        acc = np.zeros((n_classes, h, w), np.float32)
        wsum = np.zeros((h, w), np.float32)
        ramp = _ramp(tile)
        for y, x in _tiles(h, w, tile, overlap):
            crop = img[y:y + tile, x:x + tile]
            ch, cw = crop.shape[:2]
            crop = cv2.copyMakeBorder(crop, 0, tile - ch, 0, tile - cw, cv2.BORDER_REFLECT)
            lg = forward_logits(model, _to_tensor(crop).to(device))[0]
            p = torch.softmax(lg, 0).cpu().numpy()[:, :ch, :cw]
            acc[:, y:y + ch, x:x + cw] += p * ramp[:ch, :cw]
            wsum[y:y + ch, x:x + cw] += ramp[:ch, :cw]
        prob = acc / np.maximum(wsum, 1e-6)

    class_map = prob.argmax(0).astype(np.uint8)
    ent = -(prob * np.log(prob + 1e-8)).sum(0) / np.log(n_classes)
    return class_map, ent.astype(np.float32)
