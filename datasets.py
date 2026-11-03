# Copyright (C) 2026 Pritam Chakraborty <pritam.23.research@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dataset loaders (local data, not part of the repository; layout in README, "Data").

Annotated clips: data/annotations/<clip>.csv + data/videos/<video> (eval_clips.yaml maps one to the other).
  Annotation file: '# key: value' header lines (at least `size: WxH` of the upright full-size frame), then CSV columns
  frame,x0,y0,x1,y1,none. Frame numbers are 1-based video frames; boxes are pixels of the upright full-size frame; a row with
  none=1 marks a frame with nothing safety-critical. A frame may have several box rows.
BDD-OIA: data/bdd_oia/lastframe/ ({split}_25k_images_actions.json, data/*.jpg), frame-level action labels (no boxes).
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ANNOTATIONS = DATA / "annotations"
VIDEOS = DATA / "videos"
BDD_OIA = DATA / "bdd_oia" / "lastframe"
# Boxes are snapped to the grid the annotation was scored on (640x360, or 360x640 portrait), rounding outward, then expressed
# as fractions; this reproduces the published evaluation exactly.
PROC_LANDSCAPE, PROC_PORTRAIT = (640, 360), (360, 640)


def eval_clips() -> dict[str, dict]:
    return yaml.safe_load((HERE / "eval_clips.yaml").read_text())["clips"]


def video_path(clip: str) -> Path:
    return VIDEOS / eval_clips()[clip]["video"]


def clip_for_video(path: Path) -> str | None:
    """Annotated clip id of a video file, or None."""
    path = Path(path).resolve()
    return next((c for c, meta in eval_clips().items() if (VIDEOS / meta["video"]).resolve() == path), None)


def _header_and_rows(path: Path) -> tuple[dict, list[dict]]:
    header, lines = {}, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            key, _, value = line[1:].partition(":")
            header[key.strip()] = value.strip()
        elif line.strip():
            lines.append(line)
    return header, list(csv.DictReader(lines))


def _scale_box(box, src: tuple[int, int], dst: tuple[int, int]) -> tuple[int, int, int, int]:
    """xyxy box from (width, height) src to dst; lower bounds round down and upper bounds up, so the box still covers the object."""
    sx, sy = dst[0] / src[0], dst[1] / src[1]
    x0, y0, x1, y1 = box
    clip = lambda v, hi: int(min(max(v, 0), hi))
    return (clip(math.floor(x0 * sx + 1e-9), dst[0]), clip(math.floor(y0 * sy + 1e-9), dst[1]),
            clip(math.ceil(x1 * sx - 1e-9), dst[0]), clip(math.ceil(y1 * sy - 1e-9), dst[1]))


def clip_annotations(clip: str) -> dict[int, list[tuple[float, float, float, float]]]:
    """frame number -> boxes as fractions (x0, y0, x1, y1) of the frame; an empty list = nothing safety-critical."""
    header, rows = _header_and_rows(ANNOTATIONS / f"{clip}.csv")
    src = tuple(int(v) for v in header["size"].split("x"))
    dst = PROC_LANDSCAPE if src[0] >= src[1] else PROC_PORTRAIT
    out: dict[int, list] = {}
    for r in rows:
        boxes = out.setdefault(int(r["frame"]), [])
        if r["none"] != "1":
            b = _scale_box(tuple(int(r[k]) for k in ("x0", "y0", "x1", "y1")), src, dst)
            boxes.append((b[0] / dst[0], b[1] / dst[1], b[2] / dst[0], b[3] / dst[1]))
    return out


@dataclass(frozen=True)
class OiaFrame:
    image_path: Path
    actions: tuple[int, ...]  # multi-hot over (forward, stop/slow, left, right)
    split: str


def bdd_oia_frames(split: str = "val", root: Path = BDD_OIA) -> Iterator[OiaFrame]:
    """Labelled BDD-OIA last frames. The release mixes [F, S, L, R, confuse] and [F, S, L, R] action vectors; both are read as F, S, L, R."""
    meta = json.loads((root / f"{split}_25k_images_actions.json").read_text())
    actions = {a["id"]: a["category"] for a in meta["annotations"]}
    for img in meta["images"]:
        yield OiaFrame(root / "data" / img["file_name"], tuple(actions[img["id"]][:4]), split)
