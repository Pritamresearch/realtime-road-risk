# Copyright (C) 2026 Pritam Chakraborty <pritam.23.research@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later
"""GPU model wrappers: SegFormer-B2 (Cityscapes, 19 classes) and Depth Anything 3 Metric-Large.

Both take an RGB uint8 frame and return maps at the depth resolution, so every downstream step works on one grid.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

os.environ.setdefault("DA3_LOG_LEVEL", "WARN")  # DA3 logs every call at INFO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

CITYSCAPES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation",
              "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
CITYSCAPES_RGB = np.array([
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153), (250, 170, 30),
    (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142),
    (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)], np.uint8)
IGNORE = 255  # low-confidence segmentation pixels


def _ensure_safetensors(model_dir: Path) -> None:
    """transformers < 5 refuses torch.load of .bin checkpoints on torch < 2.6; convert once with weights_only=True."""
    st, bin_ = model_dir / "model.safetensors", model_dir / "pytorch_model.bin"
    if st.exists() or not bin_.exists():
        return
    from safetensors.torch import save_file

    state = torch.load(bin_, map_location="cpu", weights_only=True)
    save_file({k: v.contiguous() for k, v in state.items()}, str(st), metadata={"format": "pt"})


class Segmenter:
    def __init__(self, model_dir: str | Path, short_side: int = 512):
        from transformers import SegformerForSemanticSegmentation

        model_dir = Path(model_dir)
        _ensure_safetensors(model_dir)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_dir, use_safetensors=True).to(DEVICE).eval().half()
        self.short_side = short_side
        self.mean, self.std = IMAGENET_MEAN.to(DEVICE), IMAGENET_STD.to(DEVICE)

    def __call__(self, rgb: np.ndarray, out_hw: tuple[int, int], min_conf: float = 0.0) -> np.ndarray:
        """Cityscapes train-id map (uint8) at out_hw; pixels whose top softmax probability is below min_conf get IGNORE."""
        cls, prob = self.predict(rgb, out_hw)
        if min_conf > 0:
            cls[prob < min_conf] = IGNORE
        return cls

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray, out_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """(class map uint8, top softmax probability float32), both at out_hw."""
        h, w = rgb.shape[:2]
        s = self.short_side / min(h, w)
        size = (max(32, round(h * s / 32) * 32), max(32, round(w * s / 32) * 32))
        x = torch.from_numpy(rgb).to(DEVICE).permute(2, 0, 1)[None].float() / 255.0
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        x = ((x - self.mean) / self.std).half()
        logits = self.model(pixel_values=x).logits
        logits = F.interpolate(logits.float(), size=out_hw, mode="bilinear", align_corners=False)
        prob, cls = logits.softmax(1).max(1)
        return cls[0].to(torch.uint8).cpu().numpy(), prob[0].cpu().numpy()


def _cache_pos_embed(model: torch.nn.Module) -> None:
    """DINOv2 re-interpolates its position embedding (bicubic, ~45 ms on a laptop GPU) on every forward; the input size of a
    video never changes, so memoise it per (tokens, w, h, dtype). Inference only: pos_embed is frozen."""
    for m in model.modules():
        if hasattr(m, "interpolate_pos_encoding") and hasattr(m, "pos_embed"):
            original, cache = m.interpolate_pos_encoding, {}

            def cached(x, w, h, _orig=original, _cache=cache):
                key = (x.shape[1], w, h, x.dtype)
                if key not in _cache:
                    _cache[key] = _orig(x, w, h)
                return _cache[key]

            m.interpolate_pos_encoding = cached


def da3_resize(rgb: np.ndarray, process_res: int) -> np.ndarray:
    """DA3's upper_bound_resize, step for step (InputProcessor): longest side to process_res, then each side to the nearest
    multiple of 14 by a second small resize. Collapsing it into one resize shifts the depth by ~2% (measured), so keep both."""
    import cv2

    h, w = rgb.shape[:2]
    s = process_res / max(h, w)
    if max(h, w) != process_res:
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC if s > 1.0 else cv2.INTER_AREA)
    h, w = rgb.shape[:2]
    near = lambda x: (x // 14) * 14 + 14 if abs((x // 14) * 14 + 14 - x) <= abs(x - (x // 14) * 14) else (x // 14) * 14
    nh, nw = max(1, near(h)), max(1, near(w))
    if (nh, nw) != (h, w):
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC if nw > w or nh > h else cv2.INTER_AREA)
    return rgb


class MetricDepth:
    def __init__(self, model_dir: str | Path, process_res: int = 504, half_backbone: bool = True):
        from depth_anything_3.api import DepthAnything3

        self.api = DepthAnything3.from_pretrained(str(model_dir)).to(DEVICE).eval()
        # backbone weights stored in the autocast dtype: otherwise every forward recasts all its fp32 weights (~40 ms).
        # The DPT head stays fp32: DA3 runs it with autocast disabled on purpose.
        if DEVICE == "cuda" and half_backbone:
            self.api.model.backbone.to(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        _cache_pos_embed(self.api)
        self.process_res = process_res
        self.mean, self.std = IMAGENET_MEAN.to(DEVICE), IMAGENET_STD.to(DEVICE)

    @torch.inference_mode()
    def __call__(self, rgb: np.ndarray, hfov_deg: float) -> tuple[np.ndarray, float]:
        """(depth in meters at the processing resolution, focal length in pixels at that resolution).

        Same preprocessing as DepthAnything3.inference (upper_bound_resize, ImageNet normalisation) but done in one resize and on
        the GPU. One image per call: DA3 cross-attends between the images of a batch, which would couple frames' depths.
        """
        small = da3_resize(rgb, self.process_res)
        h, w = small.shape[:2]
        x = torch.from_numpy(small).to(DEVICE).permute(2, 0, 1)[None].float() / 255.0
        x = ((x - self.mean) / self.std)[None]  # (B=1, N=1, 3, H, W)
        out = self.api.forward(x, None, None, [], False, False, "first")
        raw = out["depth"].float().reshape(h, w).cpu().numpy()
        f_px = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
        return (f_px * raw / 300.0).astype(np.float32), f_px


class Detector:
    """YOLO26s-seg instances painted into a label map on the depth grid (0 = background, k = instance k-1, low confidence first)."""

    def __init__(self, weights: str | Path, conf: float = 0.10):
        from ultralytics import YOLO

        self.model, self.conf = YOLO(str(weights)), conf

    def __call__(self, bgr: np.ndarray, grid_hw: tuple[int, int]):
        """(label map, COCO classes, confidences, boxes as fractions xyxy); the frame is seen at 640 px on its long side."""
        import cv2

        s = 640 / max(bgr.shape[:2])
        small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        r = self.model.predict(small, imgsz=640, conf=self.conf, retina_masks=True, verbose=False,
                               quantize=16 if DEVICE == "cuda" else None)[0]
        if r.masks is None:
            return np.zeros(grid_hw, np.uint8), np.zeros(0, np.int16), np.zeros(0, np.float32), np.zeros((0, 4), np.float32)
        h, w = r.orig_shape
        conf = r.boxes.conf.cpu().numpy()
        order = np.argsort(conf)
        masks = r.masks.data.cpu().numpy().astype(bool)
        label = np.zeros((h, w), np.uint8)
        for rank, j in enumerate(order):
            label[masks[j]] = rank + 1
        boxes = r.boxes.xyxy.cpu().numpy()[order] / np.array([w, h, w, h])
        label = cv2.resize(label, (grid_hw[1], grid_hw[0]), interpolation=cv2.INTER_NEAREST)
        return label, r.boxes.cls.cpu().numpy().astype(np.int16)[order], conf[order].astype(np.float32), boxes.astype(np.float32)


def gpu_name() -> str:
    if DEVICE != "cuda":
        return "CPU"
    return torch.cuda.get_device_name(0).replace("NVIDIA ", "").replace("GeForce ", "")


def sync() -> None:
    if DEVICE == "cuda":
        torch.cuda.synchronize()
