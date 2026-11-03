# Copyright (C) 2026 Pritam Chakraborty <pritam.23.research@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later
"""RiskPipeline: one video frame (BGR, any size) in; metric depth, segmentation, detections, risk decision and dashboard out.

    from pipeline import RiskPipeline
    pipe = RiskPipeline()                      # loads DA3METRIC-LARGE, SegFormer-B2, YOLO26s-seg once
    result = pipe(frame_bgr)                   # perceive (GPU) + decide (fusion)
    print(result.to_dict())                    # level, distances, most critical object, timings
    canvas = pipe.render(result)               # 4-panel dashboard, RGB uint8

run.py (live video) and predict_frame.py (single image) are thin wrappers around this class.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.yaml"


def resolve_weights(name: str) -> Path:
    """weights/<name> in this folder."""
    path = HERE / "weights" / name
    if path.exists():
        return path
    raise FileNotFoundError(f"weights/{name} not found: run `python download_weights.py` first")


@dataclass
class Perception:
    """GPU-stage outputs for one frame; maps live on the depth grid."""
    rgb: np.ndarray  # frame used for display (at most 1280 px)
    depth: np.ndarray  # meters
    f_px: float
    class_map: np.ndarray  # Cityscapes train ids (255 = low confidence / ego mask)
    instances: tuple | None  # (label map, COCO classes, confidences, boxes as fractions) from YOLO
    camera: dict  # hfov_deg, horizon, height_m, optional ego_mask
    timings: dict = field(default_factory=dict)  # ms
    t: float | None = None  # frame time in seconds (enables trend, TTC and alert hysteresis)


@dataclass
class FrameResult:
    perception: Perception
    decision: object  # risk.Decision
    cam: object  # risk.Camera
    alert: object = None  # hmi.Alert

    def to_dict(self) -> dict:
        """JSON-friendly summary of the decision."""
        d, p = self.decision, self.perception
        h, w = p.depth.shape

        def obj(o):
            return {"class": o.name, "distance_m": round(o.distance, 2),
                    "corridor_distance_m": None if o.corridor_distance is None else round(o.corridor_distance, 2),
                    "lateral_gap_m": round(o.gap_m, 2), "field_score": round(o.field, 4),
                    "box_xyxy_frac": [round(float(v) / s, 4) for v, s in zip(o.box, (w, h, w, h))]}

        return {"alert": self.alert.to_dict() if self.alert else None, "level": d.level, "level_distance_m": None if d.distance is None else round(d.distance, 2),
                "risk_score": round(d.score, 4), "most_critical": obj(d.top) if d.top else None,
                "zones": {z.name: {"pixels": z.px, "min_m": None if z.min_m is None else round(z.min_m, 2)} for z in d.zones},
                "objects": [obj(o) for o in sorted(d.obstacles, key=lambda o: o.distance)],
                "camera": {k: v for k, v in p.camera.items() if k != "ego_mask"} | {"f_px": round(p.f_px, 1)},
                "timings_ms": {k: round(v, 1) for k, v in p.timings.items()}}


class RiskPipeline:
    def __init__(self, config: str | Path = CONFIG, hq: bool = False, camera: dict | None = None, view: str = "research"):
        """camera: overrides for hfov_deg / horizon / height_m (e.g. from a calibration), applied to every frame.
        view: dashboard layout, "research" (4 panels) or "driver" (camera + driver alert)."""
        from models import Detector, MetricDepth, Segmenter

        self.cfg = yaml.safe_load(Path(config).read_text())
        m = self.cfg["models"]
        if hq:
            m.update(m["hq"])
        self.risk_cfg = self.cfg["risk"]
        self.camera_override = {k: v for k, v in (camera or {}).items() if v is not None}
        self.view = view
        self.segmenter = Segmenter(resolve_weights(m["segformer"]), m["segformer_short_side"])
        self.depth_model = MetricDepth(resolve_weights(m["depth"]), m["depth_process_res"])
        self.detector = Detector(resolve_weights(m["yolo"]), self.risk_cfg["yolo_conf"]) if self.risk_cfg["source"] == "yolo" else None
        from hmi import AlertTracker

        self.alerts = AlertTracker(self.risk_cfg, self.cfg["hmi"])
        self._ego_cache: dict = {}
        self._dashboard = None

    # ---- camera ---------------------------------------------------------------------------------------------------------
    def camera_params(self, source_name: str, shape) -> dict:
        """Default (landscape) or portrait preset, then per-source entry in config.yaml, then constructor overrides."""
        cam = dict(self.cfg["camera"]["portrait" if shape[0] > shape[1] else "default"])
        cam.update(self.cfg["camera"].get("per_source", {}).get(source_name, {}))
        cam.update(self.camera_override)
        return cam

    def _ego_mask(self, polys: list, shape) -> np.ndarray | None:
        if not polys:
            return None
        key = (id(polys), shape)
        if key not in self._ego_cache:
            m = np.zeros(shape, np.uint8)
            cv2.fillPoly(m, [np.round(np.array(p) * (shape[1], shape[0])).astype(np.int32) for p in polys], 1)
            self._ego_cache[key] = m.astype(bool)
        return self._ego_cache[key]

    # ---- stages ---------------------------------------------------------------------------------------------------------
    def perceive(self, bgr: np.ndarray, source_name: str = "", t: float | None = None) -> Perception:
        """GPU stage: metric depth, segmentation and detections for one BGR frame. t: frame time in seconds, for video."""
        from models import sync

        if max(bgr.shape[:2]) > 1280:
            s = 1280 / max(bgr.shape[:2])
            bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cam = self.camera_params(source_name, rgb.shape)
        t0 = time.perf_counter()
        depth, f_px = self.depth_model(rgb, cam["hfov_deg"])
        t1 = time.perf_counter()
        class_map = self.segmenter(rgb, depth.shape, self.risk_cfg["min_seg_conf"])
        t2 = time.perf_counter()
        inst = self.detector(bgr, depth.shape) if self.detector else None
        sync()
        t3 = time.perf_counter()
        return Perception(rgb, depth, f_px, class_map, inst, cam,
                          {"depth": 1e3 * (t1 - t0), "seg": 1e3 * (t2 - t1), "det": 1e3 * (t3 - t2)}, t)

    def decide(self, p: Perception) -> FrameResult:
        """CPU stage: candidates (ego mask applied), corridor geometry and the risk decision."""
        from models import IGNORE
        from risk import assess, camera_for, instance_candidates, seg_candidates

        t0 = time.perf_counter()
        r = self.risk_cfg
        ego = self._ego_mask(p.camera.get("ego_mask", []), p.depth.shape)
        if ego is not None:
            p.class_map[ego] = IGNORE
        if p.instances is not None:
            label = p.instances[0].copy()
            if ego is not None:
                label[ego] = 0
            cands = instance_candidates(label, p.instances[1], p.instances[2], p.instances[3], r["class_weights"], r["yolo_conf"], r["min_object_px"])
        else:
            cands = seg_candidates(p.class_map, r["class_weights"], r["min_object_px"])
        cam = camera_for(p.depth.shape, p.f_px, p.camera["horizon"], p.camera["height_m"])
        dec = assess(cands, p.depth, cam, r)
        alert = self.alerts.update(dec, p.t)
        p.timings["fusion"] = 1e3 * (time.perf_counter() - t0)
        return FrameResult(p, dec, cam, alert)

    def __call__(self, bgr: np.ndarray, source_name: str = "", t: float | None = None) -> FrameResult:
        """One frame. For a video pass t (seconds) on every call so the alert gets trend, TTC and hysteresis."""
        return self.decide(self.perceive(bgr, source_name, t))

    # ---- display --------------------------------------------------------------------------------------------------------
    def params_text(self, result: FrameResult) -> str:
        r, cam = self.risk_cfg, result.perception.camera
        rule = "nearest in corridor" if r["rule"] == "corridor" else f"field l={r['field_lambda_m']:g} s={r['field_sigma_m']:g}"
        return (f"{r['source']} | {rule} | corridor +/-{r['corridor_half_width_m']:.1f} m | "
                f"hfov {cam['hfov_deg']:.0f} deg (f={result.perception.f_px:.0f} px)")

    def render(self, result: FrameResult, info=None, name: str = "frame") -> np.ndarray:
        """RGB dashboard canvas. info: dashboard.Info (header, timings, reference line); a minimal one is built if omitted."""
        from dashboard import DIM, Dashboard, Info
        from models import gpu_name

        if self._dashboard is None:
            d = self.cfg["display"]
            self._dashboard = Dashboard(d["width"], d["height"], self.risk_cfg, d["history"], self.view, d["bev_range_m"])
        if info is None:
            info = Info(name, 1, None, False, 0.0, gpu_name(), dict(result.perception.timings), self.params_text(result),
                        "REF: none (single frame)", DIM, [], result.perception.camera.get("ego_mask", []))
        return self._dashboard.render(result.perception.rgb, result.perception.class_map, result.perception.depth,
                                      result.decision, result.cam, info, result.alert)

