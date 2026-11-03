"""Late fusion of obstacle candidates and metric depth into a risk decision.

Geometry: pinhole camera, principal point at the image centre column and the horizon row, zero pitch and roll. A pixel (u, v)
with depth Z has lateral offset X = (u - cx) * Z / f and height above a flat road h - (v - cy) * Z / f. The ego corridor is the
box |X| <= half width, 0 to max_height above the road, Z <= max range.

Candidates come from either source: connected components of the SegFormer obstacle classes, or YOLO26-seg instances.
Two outputs:
  level     NEAR / MID / FAR / CLEAR: the nearest distance zone holding >= min_px in-corridor obstacle pixels (the alarm).
  top       the single most critical object, by rule "corridor" (nearest in-corridor object) or "field"
            (argmax of w * exp(-d / lambda) * exp(-gap^2 / (2 sigma^2)), gap = lateral distance outside the corridor).
  score     frame risk score in [0, 1]: corridor rule clip((mid - d) / (mid - critical)); field rule the top object's field value
            with weights scaled so the largest class weight is 1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

LEVELS = ("NEAR", "MID", "FAR", "CLEAR")
CITYSCAPES_OBSTACLES = {11: "person", 12: "rider", 13: "car", 14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle"}
COCO_OBSTACLES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 6: "train", 7: "truck", 15: "cat", 16: "dog"}


@dataclass
class Camera:
    f_px: float
    cx: float
    cy: float
    height_m: float


@dataclass
class Candidate:
    name: str
    weight: float
    mask: np.ndarray  # bool at depth resolution
    box: tuple[float, float, float, float]  # xyxy at depth resolution (YOLO box, or the component's bounding box)
    conf: float = 1.0


@dataclass
class Obstacle:
    name: str
    weight: float
    box: tuple[float, float, float, float]
    area: int
    distance: float  # dist_q percentile over the object
    gap_m: float  # lateral distance outside the corridor (0 = overlaps it)
    field: float  # w/w_max * exp(-d/lambda) * lateral fall-off
    corridor_px: int
    corridor_distance: float | None
    point: tuple[int, int]  # (x, y) of the pixel the distance refers to


@dataclass
class Zone:
    name: str
    lo: float
    hi: float
    px: int = 0
    min_m: float | None = None


@dataclass
class Decision:
    level: str
    distance: float | None  # corridor distance behind the level
    score: float
    top: Obstacle | None  # most critical object (selection rule)
    zones: list[Zone]
    obstacles: list[Obstacle]
    corridor_px: int
    corridor: np.ndarray = field(repr=False)


def camera_for(shape: tuple[int, int], f_px: float, horizon: float, height_m: float) -> Camera:
    h, w = shape
    return Camera(f_px, w / 2.0, horizon * h, height_m)


def lateral_map(depth: np.ndarray, cam: Camera) -> np.ndarray:
    u = np.arange(depth.shape[1], dtype=np.float32)[None, :]
    return (u - cam.cx) * depth / cam.f_px


def corridor_mask(depth: np.ndarray, cam: Camera, half_width: float, max_range: float, max_height: float,
                  x: np.ndarray | None = None) -> np.ndarray:
    """3D corridor box: |X| <= half_width, Z <= max_range, and -0.5 m to max_height above the (flat) road."""
    x = lateral_map(depth, cam) if x is None else x
    v = np.arange(depth.shape[0], dtype=np.float32)[:, None]
    above_road = cam.height_m - (v - cam.cy) * depth / cam.f_px
    return (np.abs(x) <= half_width) & (depth > 0.3) & (depth <= max_range) & (above_road <= max_height) & (above_road >= -0.5)


def _erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return mask
    return cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=iterations).astype(bool)


def _robust(depth: np.ndarray, mask: np.ndarray, q: float) -> tuple[float, tuple[int, int]]:
    """q-th percentile of depth over mask, and the (x, y) of the mask pixel whose depth is closest to it."""
    ys, xs = np.nonzero(mask)
    vals = depth[ys, xs]
    d = float(np.percentile(vals, q))
    i = int(np.argmin(np.abs(vals - d)))
    return d, (int(xs[i]), int(ys[i]))


def seg_candidates(class_map: np.ndarray, weights: dict[str, float], min_px: int) -> list[Candidate]:
    """Connected components of the weighted Cityscapes obstacle classes."""
    out = []
    for cls, name in CITYSCAPES_OBSTACLES.items():
        if name not in weights:
            continue
        mask = class_map == cls
        if mask.sum() < min_px:
            continue
        n, comp, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] >= min_px:
                x, y, bw, bh = stats[k, :4]
                out.append(Candidate(name, weights[name], comp == k, (float(x), float(y), float(x + bw), float(y + bh))))
    return out


def instance_candidates(label: np.ndarray, classes: np.ndarray, conf: np.ndarray, boxes_frac: np.ndarray,
                        weights: dict[str, float], conf_thr: float, min_px: int) -> list[Candidate]:
    """YOLO instances (label map on the depth grid, k = instance k-1) of weighted COCO classes at or above conf_thr."""
    h, w = label.shape
    out = []
    for k in range(len(classes)):
        name = COCO_OBSTACLES.get(int(classes[k]))
        if name not in weights or conf[k] < conf_thr:
            continue
        mask = label == k + 1
        if mask.sum() < min_px:
            continue
        b = boxes_frac[k]
        out.append(Candidate(name, weights[name], mask, (b[0] * w, b[1] * h, b[2] * w, b[3] * h), float(conf[k])))
    return out


def assess(cands: list[Candidate], depth: np.ndarray, cam: Camera, cfg: dict) -> Decision:
    hw, q, min_px = cfg["corridor_half_width_m"], cfg["quantile"], cfg["min_px"]
    x = lateral_map(depth, cam)
    corridor = corridor_mask(depth, cam, hw, cfg["max_range_m"], cfg["max_height_m"], x)
    ground_row = cam.cy + cfg["grounded_tolerance"] * depth.shape[0]
    zones = [Zone("NEAR", 0.0, cfg["near_m"]), Zone("MID", cfg["near_m"], cfg["mid_m"]), Zone("FAR", cfg["mid_m"], cfg["max_range_m"])]
    lam, sigma, dq = cfg["field_lambda_m"], cfg["field_sigma_m"], cfg["dist_q"]
    w_max = max(cfg["class_weights"].values())  # fixed per config, so field scores compare across frames

    obstacles: list[Obstacle] = []
    corr_obstacle = np.zeros_like(corridor)
    for c in cands:
        core = _erode(c.mask, cfg["erode"])
        if not core.any():
            core = c.mask
        dist, point = _robust(depth, core, dq)
        gap = max(0.0, float(np.percentile(np.abs(x[core]), 10)) - hw)
        lateral = 1.0 if math.isinf(sigma) else math.exp(-gap * gap / (2 * sigma * sigma))
        grounded = c.mask.any(axis=1).nonzero()[0].max() >= ground_row  # lowest pixel below the horizon
        in_corr = core & corridor if grounded else np.zeros_like(core)
        n_corr = int(in_corr.sum())
        corr_obstacle |= in_corr
        corr_d = None
        if n_corr >= min_px:
            corr_d, point = _robust(depth, in_corr, q)
        obstacles.append(Obstacle(c.name, c.weight, c.box, int(c.mask.sum()), dist, gap,
                                  (c.weight / w_max) * math.exp(-dist / lam) * lateral, n_corr, corr_d, point))

    zd = depth[corr_obstacle]
    for z in zones:
        sel = zd[(zd >= z.lo) & (zd < z.hi)]
        z.px = int(sel.size)
        z.min_m = float(np.percentile(sel, q)) if sel.size else None
    level, distance = "CLEAR", None
    for z in zones:
        if z.px >= min_px:
            level, distance = z.name, z.min_m
            break
    in_corr_objs = [o for o in obstacles if o.corridor_distance is not None]
    nearest = min(in_corr_objs, key=lambda o: o.corridor_distance) if in_corr_objs else None
    if nearest is not None and distance is None:  # objects pass min_px but no single zone does: report the object
        distance = nearest.corridor_distance
        level = next((z.name for z in zones if distance < z.hi), "CLEAR")

    if cfg["rule"] == "corridor":
        top = nearest
        score = 0.0 if distance is None else float(np.clip((cfg["mid_m"] - distance) / (cfg["mid_m"] - cfg["critical_m"]), 0.0, 1.0))
    else:
        top = max(obstacles, key=lambda o: (o.field, o.area)) if obstacles else None
        score = top.field if top is not None else 0.0
    return Decision(level, distance, score, top, zones, obstacles, int(corr_obstacle.sum()), corridor)
