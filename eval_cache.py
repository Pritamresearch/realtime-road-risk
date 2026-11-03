"""Run every model once on the 242 annotated own-clip frames and cache the outputs, so fusion variants can be evaluated offline.

Per frame (cache/eval/<clip>/<frame>.npz):
  raw504, raw392      DA3METRIC-LARGE raw output (meters = f_px * raw / 300) at both processing sizes (default and --fast)
  seg, seg_prob       SegFormer-B2 class map and top softmax probability on the 504 depth grid
  yolo_label          YOLO26s-seg instance label map on the 504 depth grid (0 = background, k = instance k-1; low confidence painted first)
  yolo_cls, yolo_conf, yolo_box   per instance; boxes as fractions of the frame (x0, y0, x1, y1)
  ref_box             annotated boxes as fractions (empty for "nothing safety-critical" frames)
Frames are prepared exactly as run.py does (upright, at most 1280 px); YOLO sees the frame at 640 px on the long side, as in v1.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from models import Detector, MetricDepth, Segmenter, sync
from pipeline import resolve_weights
from sources import V1, VideoSource

HERE = Path(__file__).resolve().parent
OUT = HERE / "cache" / "eval"
YOLO_CONF = 0.10  # lowest threshold of the grid; higher thresholds are applied offline


def clips() -> list[Path]:
    import sys

    sys.path.insert(0, str(V1))
    from datasets import own_clips

    sys.path.pop(0)
    return [(V1 / "real data" / rel) for rel in own_clips.VIDEOS.values()]


def main() -> None:
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    seg = Segmenter(resolve_weights(cfg["models"]["segformer"]), 512)  # 512 px as in the reported evaluation
    d504 = MetricDepth(resolve_weights(cfg["models"]["depth"]), 504)
    d392 = MetricDepth.__new__(MetricDepth)  # same loaded weights, smaller input
    d392.__dict__.update(d504.__dict__, process_res=392)
    yolo = Detector(resolve_weights(cfg["models"]["yolo"]), YOLO_CONF)
    times = {"depth504": [], "depth392": [], "seg": [], "yolo": []}
    for video in clips():
        src = VideoSource(video, annotated_only=True)
        clip = src.path.stem
        n = 0
        for frame in src.frames():
            out = OUT / clip / f"{frame.index:06d}.npz"
            if out.exists():
                continue
            bgr = frame.bgr
            if max(bgr.shape[:2]) > 1280:
                bgr = cv2.resize(bgr, None, fx=1280 / max(bgr.shape[:2]), fy=1280 / max(bgr.shape[:2]), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t0 = time.perf_counter()
            dep, f = d504(rgb, 60.0)
            raw504 = dep * 300.0 / f
            sync(); t1 = time.perf_counter()
            dep, f = d392(rgb, 60.0)
            raw392 = dep * 300.0 / f
            sync(); t2 = time.perf_counter()
            cls_map, sp = seg.predict(rgb, raw504.shape)
            sync(); t3 = time.perf_counter()
            label, ycls, yconf, ybox = yolo(bgr, raw504.shape)
            sync(); t4 = time.perf_counter()
            for k, v in zip(times, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
                times[k].append(v)
            ref = np.array(frame.ref.boxes, np.float32).reshape(-1, 4) if frame.ref else np.zeros((0, 4), np.float32)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, raw504=raw504.astype(np.float16), raw392=raw392.astype(np.float16), seg=cls_map,
                                seg_prob=(sp * 255).astype(np.uint8), yolo_label=label, yolo_cls=ycls, yolo_conf=yconf,
                                yolo_box=ybox, ref_box=ref, portrait=np.array(rgb.shape[0] > rgb.shape[1]))
            n += 1
        print(f"{clip}: {n} frames cached", flush=True)
    for k, v in times.items():
        if v:
            print(f"{k}: median {1e3 * np.median(v):.0f} ms over {len(v)} frames")


if __name__ == "__main__":
    main()
