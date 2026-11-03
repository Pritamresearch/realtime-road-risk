# Copyright (C) 2026 Pritam Chakraborty <pritam.23.research@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Frame sources: video files (rotation metadata applied), webcams, image folders and BDD-OIA.

Each yields Frame objects with an optional eval-only reference: the human "must attend to" boxes of the annotated clips
(datasets.py, 1-based video frame numbers) or the BDD-OIA action label. References are drawn and logged, never used by the risk rule.
"""
from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

import datasets

ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class Reference:
    kind: str  # "annotator" | "bdd_oia"
    boxes: list[tuple[float, float, float, float]]  # xyxy as fractions of width/height; empty = "nothing critical"
    label: str = ""


@dataclass
class Frame:
    index: int  # 1-based
    bgr: np.ndarray
    ref: Reference | None = None
    t: float | None = None  # seconds (video time, or wall clock for cameras); None for unrelated still images


class Source:
    name: str
    total: int | None
    fps: float
    live: bool  # webcam: always drop to the newest frame

    def frames(self) -> Iterator[Frame]:
        raise NotImplementedError


def _clip_refs(video: Path) -> dict[int, Reference]:
    """Annotator boxes for an annotated evaluation clip, keyed by frame number; {} for any other video."""
    try:
        clip = datasets.clip_for_video(video)
    except FileNotFoundError:  # no eval_clips.yaml
        return {}
    if clip is None or not (datasets.ANNOTATIONS / f"{clip}.csv").exists():
        return {}
    return {frame: Reference("annotator", boxes) for frame, boxes in datasets.clip_annotations(clip).items()}


class VideoSource(Source):
    def __init__(self, path: str | Path, annotated_only: bool = False, every: int = 1):
        self.path = Path(path)
        self.name = self.path.stem
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise FileNotFoundError(self.path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        cap.release()
        self.live = False
        self.refs = _clip_refs(self.path)
        self.annotated_only, self.every = annotated_only, every

    def frames(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(str(self.path))
        rotation = ROTATIONS.get(int(cap.get(cv2.CAP_PROP_ORIENTATION_META)))
        if rotation is not None:  # OpenCV builds that auto-rotate report the rotation but already applied it
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        wanted = set(self.refs) if self.annotated_only else None
        number = 0
        while True:
            if wanted is not None and not wanted:
                break
            skip = (wanted is not None and number + 1 not in wanted) or (wanted is None and number % self.every)
            number += 1
            if skip:  # sequential grab: seeking in compressed video is not frame-exact
                if not cap.grab():
                    break
                continue
            ok, bgr = cap.read()
            if not ok:
                break
            if wanted is not None:
                wanted.discard(number)
            yield Frame(number, cv2.rotate(bgr, rotation) if rotation is not None else bgr, self.refs.get(number), (number - 1) / self.fps)
        cap.release()


class CameraSource(Source):
    def __init__(self, index: int):
        self.index, self.name, self.total, self.fps, self.live = index, f"camera-{index}", None, 30.0, True

    def frames(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        number = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            number += 1
            yield Frame(number, bgr, t=time.perf_counter())
        cap.release()


class ImageFolderSource(Source):
    def __init__(self, folder: str | Path, every: int = 1):
        """A folder of images (sorted by name), or a single image file."""
        folder = Path(folder)
        self.paths = [folder] if folder.is_file() else sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)[::every]
        self.name, self.total, self.fps, self.live = folder.stem, len(self.paths), 2.0, False

    def frames(self) -> Iterator[Frame]:
        for i, p in enumerate(self.paths, 1):
            yield Frame(i, cv2.imread(str(p)))


class BddOiaSource(Source):
    """BDD-OIA last frames of one split, with the stop/slow-down vs forward label as reference."""

    def __init__(self, split: str = "val", n: int | None = 200, seed: int = 0):
        frames = list(datasets.bdd_oia_frames(split))
        if n is not None and n < len(frames):
            rng = np.random.default_rng(seed)
            frames = [frames[i] for i in sorted(rng.choice(len(frames), size=n, replace=False))]
        self.items = frames
        self.name, self.total, self.fps, self.live = f"bdd_oia-{split}", len(frames), 2.0, False

    def frames(self) -> Iterator[Frame]:
        for i, f in enumerate(self.items, 1):
            acts = [n for n, a in zip(("forward", "stop/slow", "left", "right"), f.actions) if a]
            yield Frame(i, cv2.imread(str(f.image_path)), Reference("bdd_oia", [], "+".join(acts) or "none"))


class LatestFrame:
    """Reads a source on a thread, paced at its native frame rate, keeping only the newest frame (real-time playback)."""

    def __init__(self, source: Source):
        self.source, self._frame, self._done = source, None, False
        self._cv = threading.Condition()
        self.paused = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        period, t0, k = 1.0 / max(self.source.fps, 1e-3), time.perf_counter(), 0
        for frame in self.source.frames():
            while self.paused:
                time.sleep(0.05)
                t0, k = time.perf_counter(), 0
            if not self.source.live:
                k += 1
                delay = t0 + k * period - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            with self._cv:
                self._frame = frame
                self._cv.notify_all()
        with self._cv:
            self._done = True
            self._cv.notify_all()

    def __iter__(self) -> Iterator[Frame]:
        while True:
            with self._cv:
                while self._frame is None and not self._done:
                    self._cv.wait()
                if self._frame is None:
                    return
                frame, self._frame = self._frame, None
            yield frame


def open_source(spec: str, annotated_only: bool = False, every: int = 1, n: int | None = 200) -> Source:
    """spec: a video file, an image file, an image folder, a camera index ("0"), "bdd_oia[:split]", or part of a file name
    looked up in data/videos/."""
    if spec.isdigit():
        return CameraSource(int(spec))
    if spec.startswith("bdd_oia"):
        return BddOiaSource(spec.partition(":")[2] or "val", n)
    p = Path(spec)
    if not p.exists():
        root = datasets.VIDEOS
        hits = [q for q in root.rglob("*") if q.is_file() and spec.lower() in q.name.lower()] if root.exists() else []
        if not hits:
            raise FileNotFoundError(spec)
        p = hits[0]
    if p.is_dir() or p.suffix.lower() in IMAGE_EXT:
        return ImageFolderSource(p, every)
    return VideoSource(p, annotated_only, every)
