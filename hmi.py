"""Driver-facing alert logic on top of the per-frame decision: one alert state, object trend, time to collision, hysteresis.

States (forward-collision-warning convention: colour + icon + word, never colour alone):
  WARNING  red    an obstacle in the corridor in the NEAR zone, or approaching with TTC < ttc_warning_s
  CAUTION  amber  an obstacle in the corridor in the MID zone, or approaching with TTC < ttc_caution_s
  CLEAR    grey   nothing in the corridor within the MID range (a FAR lead object is shown as information, not as "safe green")

Hysteresis: a higher state is shown at once; a lower one only after it has held for `hold_s` seconds (no flicker).

Trend and TTC come from tracking objects across frames (greedy IoU matching within a class) and fitting a line to each track's
distance over the last `window_s` seconds. A trend is reported only when the slope is significant (|slope| > 2 standard errors
and > `min_speed_mps`): monocular depth jitters by a few percent per frame, so without this test TTC would be noise.
TTC is relative (no ego speed is needed): distance / closing speed. It is an estimate and has not been validated against ground truth.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

STATES = ("CLEAR", "CAUTION", "WARNING")
RANK = {s: i for i, s in enumerate(STATES)}


@dataclass
class Trend:
    closing_mps: float | None = None  # > 0 approaching, < 0 receding; None = not significant or not measured
    ttc_s: float | None = None
    measured: bool = False  # enough samples to fit a trend (False for single images and new tracks)

    @property
    def label(self) -> str:
        if not self.measured:
            return "unknown"
        if self.closing_mps is None:
            return "steady"
        return "approaching" if self.closing_mps > 0 else "moving away"


@dataclass
class Alert:
    state: str  # shown (after hysteresis)
    raw_state: str  # this frame alone
    obj: object | None  # risk.Obstacle the alert is about: the nearest in-corridor object, else the most critical object
    distance: float | None
    bearing: str  # AHEAD, AHEAD-LEFT, AHEAD-RIGHT (in the corridor), LEFT, RIGHT (outside it)
    trend: Trend = field(default_factory=Trend)
    reason: str = ""
    watch: object | None = None  # the risk field's most critical object when it differs from obj (e.g. a pedestrian at the kerb)
    held_s: float | None = None  # seconds since obj was last the live cause (state held by hysteresis); None = live

    def to_dict(self) -> dict:
        return {"state": self.state, "raw_state": self.raw_state, "object": self.obj.name if self.obj else None,
                "distance_m": None if self.distance is None else round(self.distance, 2), "bearing": self.bearing,
                "trend": self.trend.label, "closing_speed_mps": None if self.trend.closing_mps is None else round(self.trend.closing_mps, 2),
                "ttc_s": None if self.trend.ttc_s is None else round(self.trend.ttc_s, 2), "reason": self.reason,
                "watch": None if self.watch is None else self.watch.name, "held_s": self.held_s}


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


class _Track:
    def __init__(self, obj, t: float):
        self.name, self.box = obj.name, obj.box
        self.samples: deque[tuple[float, float]] = deque(maxlen=12)
        self.add(obj, t)

    def add(self, obj, t: float) -> None:
        self.box = obj.box
        self.samples.append((t, obj.corridor_distance if obj.corridor_distance is not None else obj.distance))
        self.last_t = t


class AlertTracker:
    def __init__(self, risk_cfg: dict, hmi_cfg: dict):
        self.r, self.c = risk_cfg, hmi_cfg
        self.tracks: list[_Track] = []
        self.shown, self.lower_since, self.last = "CLEAR", None, None
        self.history: deque[str] = deque(maxlen=hmi_cfg.get("history", 60))

    def reset(self) -> None:
        self.tracks, self.shown, self.lower_since, self.last = [], "CLEAR", None, None
        self.history.clear()

    # ---- tracking -----------------------------------------------------------------------------------------------------
    def _associate(self, obstacles, t: float) -> dict[int, _Track]:
        """Greedy IoU matching within a class; returns id(obstacle) -> track."""
        self.tracks = [tr for tr in self.tracks if t - tr.last_t <= self.c["track_timeout_s"]]
        pairs = sorted(((_iou(o.box, tr.box), i, j) for i, o in enumerate(obstacles) for j, tr in enumerate(self.tracks)
                        if o.name == tr.name), reverse=True)
        used_o, used_t, out = set(), set(), {}
        for iou, i, j in pairs:
            if iou < self.c["match_iou"] or i in used_o or j in used_t:
                continue
            used_o.add(i), used_t.add(j)
            self.tracks[j].add(obstacles[i], t)
            out[id(obstacles[i])] = self.tracks[j]
        for i, o in enumerate(obstacles):
            if i not in used_o:
                tr = _Track(o, t)
                self.tracks.append(tr)
                out[id(o)] = tr
        return out

    def _trend(self, track: _Track | None, t: float) -> Trend:
        if track is None:
            return Trend()
        pts = [(ti, d) for ti, d in track.samples if t - ti <= self.c["window_s"]]
        if len(pts) < self.c["min_samples"]:
            return Trend()
        ts, ds = np.array(pts).T
        ts = ts - ts.mean()
        if np.ptp(ts) <= 0:
            return Trend()
        slope, icpt = np.polyfit(ts, ds, 1)
        resid = ds - (slope * ts + icpt)
        se = math.sqrt(max((resid ** 2).sum() / max(len(ts) - 2, 1), 1e-9) / (ts ** 2).sum())
        if abs(slope) < max(self.c["min_speed_mps"], 2 * se):
            return Trend(measured=True)
        closing = -float(slope)
        ttc = ds[-1] / closing if closing > 0 else None
        return Trend(closing, ttc, True)

    # ---- alert --------------------------------------------------------------------------------------------------------
    def bearing(self, obj) -> str:
        if obj is None:
            return ""
        hw = self.r["corridor_half_width_m"]
        if obj.corridor_distance is not None:  # reaches the corridor
            return "AHEAD" if abs(obj.x_m) <= hw else ("AHEAD-RIGHT" if obj.x_m > 0 else "AHEAD-LEFT")
        return "LEFT" if obj.x_m < 0 else "RIGHT"

    def update(self, decision, t: float | None) -> Alert:
        """t: frame time in seconds (None for an isolated image: no trend, no hysteresis)."""
        top = decision.top
        in_corr = [o for o in decision.obstacles if o.corridor_distance is not None]
        focus = min(in_corr, key=lambda o: o.corridor_distance) if in_corr else top
        tracks = self._associate(decision.obstacles, t) if t is not None else {}
        trend = self._trend(tracks.get(id(focus)), t) if (t is not None and focus is not None) else Trend()

        raw, reason = {"NEAR": "WARNING", "MID": "CAUTION"}.get(decision.level, "CLEAR"), f"corridor zone {decision.level}"
        if in_corr and trend.ttc_s is not None:
            if trend.ttc_s < self.c["ttc_warning_s"] and RANK[raw] < RANK["WARNING"]:
                raw, reason = "WARNING", f"time to collision {trend.ttc_s:.1f} s"
            elif trend.ttc_s < self.c["ttc_caution_s"] and RANK[raw] < RANK["CAUTION"]:
                raw, reason = "CAUTION", f"time to collision {trend.ttc_s:.1f} s"

        if t is None:
            shown = raw
        elif RANK[raw] >= RANK[self.shown]:
            shown, self.lower_since = raw, None
        else:  # lower state must hold for hold_s before it is shown
            self.lower_since = t if self.lower_since is None else self.lower_since
            shown = raw if t - self.lower_since >= self.c["hold_s"] else self.shown
            if shown == raw:
                self.lower_since = None
        self.shown = shown
        self.history.append(shown)
        distance = None if focus is None else (focus.corridor_distance if focus.corridor_distance is not None else focus.distance)
        if shown != raw and self.last is not None and self.last.obj is not None:  # holding: keep showing what caused the state
            alert = Alert(shown, raw, self.last.obj, self.last.distance, self.last.bearing, self.last.trend, "held: " + self.last.reason,
                          top if top is not self.last.obj else None, t - self.last_t)
        else:
            alert = Alert(shown, raw, focus, distance, self.bearing(focus), trend, reason, top if top is not focus else None)
        if shown == raw:
            self.last, self.last_t = alert, t
        return alert
