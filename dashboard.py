# Copyright (C) 2026 Pritam Chakraborty <pritam.23.research@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Four-panel dashboard: camera with corridor and obstacle distances, segmentation, metric depth, risk decision.

Shapes are drawn with OpenCV on an RGB canvas; text is queued and drawn in one PIL pass at the end (monospace TrueType).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from models import CITYSCAPES, CITYSCAPES_RGB
from risk import Camera, Decision

BG = (9, 12, 16)
PANEL = (13, 17, 22)
BORDER = (36, 46, 56)
TITLE_BG = (24, 31, 39)
TEXT = (214, 222, 230)
DIM = (120, 134, 146)
ACCENT = (64, 200, 222)
LEVEL_RGB = {"NEAR": (236, 72, 64), "MID": (240, 168, 40), "FAR": (96, 150, 186), "CLEAR": (120, 134, 146)}  # FAR is neutral, never "safe green"
# driver alert: colour + icon + word together
ALERT_FILL = {"WARNING": (206, 40, 36), "CAUTION": (242, 168, 28), "CLEAR": (22, 29, 37)}
ALERT_INK = {"WARNING": (255, 255, 255), "CAUTION": (24, 18, 6), "CLEAR": (214, 222, 230)}
ALERT_MARK = {"WARNING": (236, 72, 64), "CAUTION": (240, 168, 40), "CLEAR": (176, 186, 196)}
HEADER_H, GAP, TITLE_H = 34, 6, 24
PALETTE = np.zeros((256, 3), np.uint8)  # class id -> RGB; IGNORE (255) and unused ids stay black
PALETTE[:len(CITYSCAPES_RGB)] = CITYSCAPES_RGB

_FONT_FILES = {False: ["consola.ttf", "DejaVuSansMono.ttf", "cour.ttf"], True: ["consolab.ttf", "DejaVuSansMono-Bold.ttf", "courbd.ttf"]}
_fonts: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}


def font(size: int, bold: bool = False):
    key = (size, bold)
    if key not in _fonts:
        for name in _FONT_FILES[bold]:
            for path in (name, str(Path("C:/Windows/Fonts") / name)):
                try:
                    _fonts[key] = ImageFont.truetype(path, size)
                    break
                except OSError:
                    continue
            if key in _fonts:
                break
        else:
            _fonts[key] = ImageFont.load_default(size)
    return _fonts[key]


@dataclass
class Info:
    """Everything the dashboard shows besides the images and the decision."""
    source: str
    frame: int
    total: int | None
    live: bool
    fps: float
    gpu: str
    timings: dict[str, float]  # ms
    params: str
    ref_text: str
    ref_color: tuple[int, int, int]
    ref_boxes: list[tuple[float, float, float, float]]
    ego_polys: list = None  # ego-vehicle mask polygons, fractions of the frame


def object_distance(o) -> float:
    """In-corridor distance when the object reaches the corridor, else its own distance."""
    return o.corridor_distance if o.corridor_distance is not None else o.distance


class Dashboard:
    def __init__(self, width: int, height: int, risk_cfg: dict, history: int = 60, view: str = "research", bev_range_m: float = 45.0):
        """view: "research" (camera, segmentation, depth, driver alert) or "driver" (camera + driver alert only)."""
        self.w, self.h, self.cfg, self.view, self.bev_range = width, height, risk_cfg, view, bev_range_m
        self.history: deque[str] = deque(maxlen=history)
        if view == "driver":
            cw = int((width - 3 * GAP) * 0.64)
            ph = height - HEADER_H - 2 * GAP
            self.panels = [(GAP, HEADER_H + GAP, cw, ph), (2 * GAP + cw, HEADER_H + GAP, width - 3 * GAP - cw, ph)]
        else:
            pw, ph = (width - 3 * GAP) // 2, (height - HEADER_H - 3 * GAP) // 2
            self.panels = [(GAP + c * (pw + GAP), HEADER_H + GAP + r * (ph + GAP), pw, ph) for r in range(2) for c in range(2)]
        self.texts: list[tuple] = []

    def zone_of(self, d: float) -> str:
        return "NEAR" if d < self.cfg["near_m"] else "MID" if d < self.cfg["mid_m"] else "FAR"

    # ---- text queue -------------------------------------------------------------------------------------------------
    def text(self, x, y, s, size=13, color=TEXT, bold=False, anchor="la", bg=None):
        self.texts.append((x, y, s, size, color, bold, anchor, bg))

    def _flush_text(self, canvas: np.ndarray) -> np.ndarray:
        img = Image.fromarray(canvas)
        d = ImageDraw.Draw(img)
        for x, y, s, size, color, bold, anchor, bg in self.texts:
            f = font(size, bold)
            if bg is not None:
                l, t, r, b = d.textbbox((x, y), s, font=f, anchor=anchor)
                d.rectangle((l - 3, t - 2, r + 3, b + 2), fill=bg)
            d.text((x, y), s, font=f, fill=color, anchor=anchor)
        self.texts = []
        return np.asarray(img)

    # ---- helpers ----------------------------------------------------------------------------------------------------
    def _panel_frame(self, canvas, i: int, title: str, subtitle: str):
        """Draw panel i's frame and title bar; returns the content rect."""
        x, y, w, h = self.panels[i]
        cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), PANEL, -1)
        cv2.rectangle(canvas, (x, y), (x + w - 1, y + TITLE_H), TITLE_BG, -1)
        cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), BORDER, 1)
        cv2.rectangle(canvas, (x + 1, y + 1), (x + 26, y + TITLE_H - 1), (20, 70, 84), -1)
        self.text(x + 13, y + TITLE_H // 2 + 1, f"{i + 1:02d}", 12, ACCENT, True, "mm")
        self.text(x + 34, y + TITLE_H // 2 + 1, title, 12, TEXT, True, "lm")
        if subtitle:
            tw = font(12, True).getlength(title)
            self.text(x + 34 + tw + 10, y + TITLE_H // 2 + 1, "|  " + subtitle, 12, DIM, False, "lm")
        return x + 1, y + TITLE_H + 1, w - 2, h - TITLE_H - 2

    @staticmethod
    def _fit(rect, ih, iw):
        x, y, w, h = rect
        s = min(w / iw, h / ih)
        dw, dh = int(round(iw * s)), int(round(ih * s))
        return x + (w - dw) // 2, y + (h - dh) // 2, dw, dh

    @staticmethod
    def _blit(canvas, img, fit):
        x, y, w, h = fit
        canvas[y:y + h, x:x + w] = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if img.shape[1] > w else cv2.INTER_LINEAR)

    @staticmethod
    def _dashed_rect(canvas, p0, p1, color, dash=7):
        (x0, y0), (x1, y1) = p0, p1
        for a, b, horiz, c in ((x0, x1, True, y0), (x0, x1, True, y1), (y0, y1, False, x0), (y0, y1, False, x1)):
            for s in range(a, b, dash * 2):
                e = min(s + dash, b)
                cv2.line(canvas, (s, c) if horiz else (c, s), (e, c) if horiz else (c, e), color, 1, cv2.LINE_AA)

    # ---- panels -----------------------------------------------------------------------------------------------------
    def _camera(self, canvas, rgb, dec: Decision, cam: Camera, depth_hw, info: Info, alert):
        rect = self._panel_frame(canvas, 0, "CAMERA", "RISK ZONES + OBSTACLE DISTANCE")
        fx, fy, fw, fh = fit = self._fit(rect, *rgb.shape[:2])
        self._blit(canvas, rgb, fit)
        dh, dw = depth_hw
        sx, sy = fw / dw, fh / dh
        to_disp = lambda u, v: (int(round(fx + u * sx)), int(round(fy + v * sy)))

        # ground-plane corridor (flat road, zero pitch): ground point at (X, Z) -> u = cx + f X / Z, v = cy + f H / Z
        c = self.cfg
        hw = c["corridor_half_width_m"]
        z_bottom = cam.f_px * cam.height_m / max(dh - cam.cy, 1.0)
        x0, y0, w0, h0 = fit
        roi = (slice(y0, y0 + h0), slice(x0, x0 + w0))
        img = canvas[roi].copy()  # draw in image coordinates so the corridor is clipped to the frame, not the letterbox
        overlay = img.copy()
        to_disp = lambda u, v: (int(round(u * sx)), int(round(v * sy)))
        edges = []
        for name, z0, z1 in (("NEAR", z_bottom, c["near_m"]), ("MID", c["near_m"], c["mid_m"]), ("FAR", c["mid_m"], c["max_range_m"])):
            z0 = max(z0, z_bottom)
            if z1 <= z0:
                continue
            pts = []
            for z, side in ((z0, -1), (z1, -1), (z1, 1), (z0, 1)):
                pts.append(to_disp(cam.cx + side * cam.f_px * hw / z, cam.cy + cam.f_px * cam.height_m / z))
            pts = np.array(pts, np.int32)
            cv2.fillPoly(overlay, [pts], LEVEL_RGB[name])
            edges.append((pts, LEVEL_RGB[name]))
        img = cv2.addWeighted(overlay, 0.32, img, 0.68, 0)
        for pts, col in edges:
            cv2.polylines(img, [pts], True, col, 1, cv2.LINE_AA)
        canvas[roi] = img
        to_disp = lambda u, v: (int(round(fx + u * sx)), int(round(fy + v * sy)))
        for z in (c["near_m"], c["mid_m"]):
            u, v = to_disp(cam.cx - cam.f_px * hw / z, cam.cy + cam.f_px * cam.height_m / z)
            if fy < v < fy + fh and u - 40 > fx:
                self.text(u - 6, v, f"{z:.0f} m", 12, TEXT, False, "rm")

        # objects; the one the alert is about is highlighted in the alert colour
        for o in sorted(dec.obstacles, key=lambda o: -o.distance):
            p0, p1 = to_disp(o.box[0], o.box[1]), to_disp(o.box[2], o.box[3])
            is_top = alert.obj is o
            if is_top:
                col = ALERT_MARK[alert.state]
            elif o.corridor_distance is not None:
                col = (200, 200, 200)
            else:
                col = (150, 156, 162)
            cv2.rectangle(canvas, p0, p1, col, 2 if is_top else 1, cv2.LINE_AA)
            label = f"{o.name} {object_distance(o) if is_top else o.distance:.1f}m"
            lx = min(max(p0[0], fx + 3), fx + fw - 3 - int(font(12, is_top).getlength(label)))
            ly = max(p0[1] - 2, fy + 16)
            self.text(lx, ly, label, 12, (10, 14, 18) if is_top else TEXT, is_top, "lb", bg=col if is_top else (40, 46, 52))

        for poly in info.ego_polys or []:
            pts = np.array([(fx + px * fw, fy + py * fh) for px, py in poly], np.int32)
            cv2.polylines(canvas, [pts], True, (110, 120, 130), 1, cv2.LINE_AA)
        for b in info.ref_boxes:
            self._dashed_rect(canvas, (int(fx + b[0] * fw), int(fy + b[1] * fh)), (int(fx + b[2] * fw), int(fy + b[3] * fh)), (255, 255, 255))
        if info.ref_boxes:
            self.text(int(fx + info.ref_boxes[0][0] * fw), int(fy + info.ref_boxes[0][3] * fh) + 2, "REF", 11, (255, 255, 255), True, "la", bg=(60, 60, 60))

        badge = alert.state + (f"  {alert.obj.name} {alert.distance:.1f} m {alert.bearing.lower()}" if alert.obj is not None else "")
        if alert.held_s is not None:
            badge = f"{alert.state}  (holding, last {alert.obj.name} {alert.distance:.1f} m)"
        self.text(rect[0] + 10, rect[1] + rect[3] - 10, badge, 18, ALERT_INK[alert.state], True, "lb",
                  bg=ALERT_FILL[alert.state] if alert.state != "CLEAR" else (40, 48, 58))

    def _segmentation(self, canvas, rgb, class_map):
        rect = self._panel_frame(canvas, 1, "SEMANTIC SEGMENTATION", "SegFormer-B2 / Cityscapes")
        x, y, w, h = rect
        fit = self._fit((x, y, w, h - 24), *rgb.shape[:2])
        color = PALETTE[cv2.resize(class_map, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)]
        self._blit(canvas, cv2.addWeighted(color, 0.62, rgb, 0.38, 0), fit)
        counts = np.bincount(class_map.ravel(), minlength=256)[:len(CITYSCAPES)]
        lx, ly = x + 10, y + h - 12
        for k in np.argsort(-counts)[:9]:
            if counts[k] == 0:
                continue
            cv2.rectangle(canvas, (lx, ly - 5), (lx + 10, ly + 5), tuple(int(v) for v in CITYSCAPES_RGB[k]), -1)
            cv2.rectangle(canvas, (lx, ly - 5), (lx + 10, ly + 5), BORDER, 1)
            self.text(lx + 15, ly, CITYSCAPES[k], 12, TEXT, False, "lm")
            lx += 26 + int(font(12).getlength(CITYSCAPES[k]))
            if lx > x + w - 60:
                break

    def _depth(self, canvas, depth, dec: Decision, alert):
        rect = self._panel_frame(canvas, 2, "METRIC DEPTH", "Depth Anything 3 (DA3METRIC-LARGE)")
        x, y, w, h = rect
        bar_w = 64
        fx, fy, fw, fh = fit = self._fit((x, y, w - bar_w, h), *depth.shape)
        rng = self.cfg["max_range_m"]
        norm = (255 * (1.0 - np.clip(depth / rng, 0, 1))).astype(np.uint8)
        vis = cv2.cvtColor(cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
        self._blit(canvas, vis, fit)

        # colour bar: 0 m (top, near) .. max range (bottom, far)
        bx, by0, by1 = x + w - bar_w + 14, y + 22, y + h - 22
        ramp = (255 * (1.0 - np.linspace(0, 1, by1 - by0)))[:, None].astype(np.uint8)
        ramp = cv2.cvtColor(cv2.applyColorMap(np.repeat(ramp, 14, 1), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
        canvas[by0:by1, bx:bx + 14] = ramp
        self.text(bx + 7, by0 - 6, "m", 11, DIM, False, "mb")
        for t in np.linspace(0, rng, 5):
            ty = int(by0 + (by1 - by0 - 1) * t / rng)
            cv2.line(canvas, (bx + 14, ty), (bx + 18, ty), TEXT, 1)
            self.text(bx + 21, ty, f"{t:.0f}", 11, TEXT, False, "lm")

        n = alert.obj if any(o is alert.obj for o in dec.obstacles) else None  # live objects only, never a held one
        if n is not None:
            px = int(fx + (n.point[0] + 0.5) * fw / depth.shape[1])
            py = int(fy + (n.point[1] + 0.5) * fh / depth.shape[0])
            cv2.circle(canvas, (px, py), 7, ACCENT, 2, cv2.LINE_AA)
            cv2.line(canvas, (px - 11, py), (px + 11, py), ACCENT, 1, cv2.LINE_AA)
            cv2.line(canvas, (px, py - 11), (px, py + 11), ACCENT, 1, cv2.LINE_AA)
            label = f"d= {object_distance(n):.1f} m"
            lx = min(px + 12, fx + fw - 4 - int(font(12, True).getlength(label)))
            self.text(lx, max(py - 10, fy + 18), label, 12, (8, 20, 26), True, "lb", bg=ACCENT)

    # ---- driver alert panel -------------------------------------------------------------------------------------------
    @staticmethod
    def _icon(canvas, cx: int, cy: int, size: int, state: str) -> None:
        """Warning triangle with '!' (WARNING, CAUTION) or a check in a circle (CLEAR)."""
        ink, fill = ALERT_INK[state], ALERT_FILL[state]
        if state == "CLEAR":
            r = size // 2
            cv2.circle(canvas, (cx, cy), r, ink, max(2, size // 14), cv2.LINE_AA)
            pts = np.array([(cx - r * 0.45, cy + r * 0.02), (cx - r * 0.1, cy + r * 0.38), (cx + r * 0.5, cy - r * 0.35)], np.int32)
            cv2.polylines(canvas, [pts], False, ink, max(2, size // 12), cv2.LINE_AA)
            return
        h = int(size * 0.9)
        tri = np.array([(cx, cy - h // 2), (cx - size // 2, cy + h // 2), (cx + size // 2, cy + h // 2)], np.int32)
        cv2.fillPoly(canvas, [tri], ink, cv2.LINE_AA)
        bar_w = max(3, size // 9)
        cv2.rectangle(canvas, (cx - bar_w // 2, cy - h // 5), (cx + bar_w // 2, cy + h // 6), fill, -1)
        cv2.circle(canvas, (cx, cy + h // 3), max(2, bar_w // 2 + 1), fill, -1, cv2.LINE_AA)

    @staticmethod
    def _arrow(canvas, cx: int, cy: int, size: int, bearing: str, color) -> None:
        d = {"AHEAD": (0, -1), "LEFT": (-1, 0), "RIGHT": (1, 0), "AHEAD-LEFT": (-0.7, -0.7), "AHEAD-RIGHT": (0.7, -0.7)}.get(bearing)
        if d is None:
            return
        p0 = (int(cx - d[0] * size / 2), int(cy - d[1] * size / 2))
        p1 = (int(cx + d[0] * size / 2), int(cy + d[1] * size / 2))
        cv2.arrowedLine(canvas, p0, p1, color, max(2, size // 8), cv2.LINE_AA, tipLength=0.45)

    def _alert_card(self, canvas, rect, alert) -> None:
        x, y, w, h = rect
        st, ink = alert.state, ALERT_INK[alert.state]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), ALERT_FILL[st], -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), ALERT_MARK[st] if st == "CLEAR" else ink, 1)
        s = min(w / 420, h / 250)  # type scale
        icon = int(64 * s)
        self._icon(canvas, x + int(18 * s) + icon // 2, y + int(16 * s) + icon // 2, icon, st)
        self.text(x + int(30 * s) + icon, y + int(16 * s) + icon // 2, st, int(50 * s), ink, True, "lm")
        sub = {"WARNING": "obstacle close in your path", "CAUTION": "obstacle in your path", "CLEAR": "no obstacle in your path"}[st]
        if alert.raw_state != st:
            sub += "  (holding)"
        self.text(x + int(30 * s) + icon, y + int(16 * s) + icon + int(4 * s), sub, int(15 * s), ink, False, "la")

        oy = y + int(16 * s) + icon + int(44 * s)
        o = alert.obj
        if o is None:
            self.text(x + int(18 * s), oy + int(10 * s), "nothing detected", int(22 * s), ink, False)
        elif st == "CLEAR":  # nothing to act on: nearby objects as a quiet info line, not headline numbers
            tr = alert.trend
            trend = "" if tr.closing_mps is None else (", approaching" if tr.closing_mps > 0 else ", moving away")
            where = "ahead in lane" if o.corridor_distance is not None else alert.bearing.lower()
            self.text(x + int(18 * s), oy + int(6 * s), "nearest road user", int(15 * s), DIM, False, "la")
            self.text(x + int(18 * s), oy + int(30 * s), f"{o.name} {alert.distance:.1f} m {where}{trend}", int(22 * s), ink, True, "la")
        elif alert.held_s is not None:  # hysteresis hold: the cause is no longer live, say so instead of showing stale numbers
            self.text(x + int(18 * s), oy + int(6 * s), f"last seen {alert.held_s:.1f} s ago:", int(18 * s), ink, False, "la")
            self.text(x + int(18 * s), oy + int(34 * s), f"{o.name} {alert.distance:.1f} m {alert.bearing.lower()}", int(26 * s), ink, True, "la")
        else:
            num = f"{alert.distance:.1f}" if alert.distance < 100 else ">100"
            self.text(x + int(18 * s), oy, num, int(58 * s), ink, True, "la")
            ux = x + int(22 * s) + int(font(int(58 * s), True).getlength(num))
            self.text(ux, oy + int(22 * s), "m", int(28 * s), ink, True, "la")
            dx = ux + int(26 * s) + int(font(int(28 * s), True).getlength("m"))
            self.text(dx, oy + int(6 * s), o.name.upper(), int(24 * s), ink, True, "la")
            self._arrow(canvas, dx + int(12 * s), oy + int(50 * s), int(22 * s), alert.bearing, ink)
            self.text(dx + int(30 * s), oy + int(50 * s), alert.bearing.replace("-", " "), int(18 * s), ink, True, "lm")
            tr = alert.trend
            if not tr.measured:
                trend = "trend not yet measured"
            elif tr.closing_mps is None:
                trend = "steady"
            elif tr.closing_mps > 0:
                in_path = o.corridor_distance is not None  # TTC only for objects in the path; beside it, closing is just ego motion
                trend = f"approaching {tr.closing_mps:.1f} m/s" + (f"   TTC {tr.ttc_s:.1f} s" if in_path and tr.ttc_s is not None and tr.ttc_s < 10 else "")
            else:
                trend = f"moving away {-tr.closing_mps:.1f} m/s"
            self.text(x + int(18 * s), oy + int(84 * s), trend, int(18 * s), ink, False, "la")
        if alert.watch is not None and alert.watch is not o:
            wd = alert.watch.corridor_distance if alert.watch.corridor_distance is not None else alert.watch.distance
            self.text(x + int(18 * s), y + h - int(12 * s), f"also watch: {alert.watch.name} {wd:.1f} m", int(15 * s), ink, False, "lb")

    def _bev(self, canvas, rect, dec: Decision, cam: Camera, alert) -> None:
        """Top view in metres: ego car at the bottom, corridor, range lines, camera field of view, objects at (X, Z)."""
        x, y, w, h = rect
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (11, 14, 18), -1)
        c, R = self.cfg, self.bev_range
        ppm = (h - 8) / (R + 6.0)  # depth scale
        xpm = min(w / 20.0, 3.0 * ppm)  # lateral scale: about +/-10 m across the panel (stretched, as ADAS top views usually are)
        ox, oy = x + w // 2, y + h - 4 - int(6.0 * ppm)  # screen point of (X=0, Z=0): the ego front bumper
        P = lambda X, Z: (int(round(ox + X * xpm)), int(round(oy - Z * ppm)))
        roi = (slice(y, y + h), slice(x, x + w))
        sub = canvas[roi].copy()
        Q = lambda X, Z: (P(X, Z)[0] - x, P(X, Z)[1] - y)

        half_fov = math.atan(cam.cx / cam.f_px)
        zf = R + 6
        wedge = np.array([Q(0, 0), Q(-math.tan(half_fov) * zf, zf), Q(math.tan(half_fov) * zf, zf)], np.int32)
        cv2.fillPoly(sub, [wedge], (17, 22, 28))
        hw = c["corridor_half_width_m"]
        overlay = sub.copy()
        for name, z0, z1 in (("NEAR", 0, c["near_m"]), ("MID", c["near_m"], c["mid_m"]), ("FAR", c["mid_m"], R)):
            cv2.rectangle(overlay, Q(-hw, z1), Q(hw, z0), LEVEL_RGB[name], -1)
        sub = cv2.addWeighted(overlay, 0.22, sub, 0.78, 0)
        for X in (-hw, hw):
            cv2.line(sub, Q(X, 0), Q(X, R), (70, 82, 94), 1, cv2.LINE_AA)
        for Z in (c["near_m"], c["mid_m"]):
            a = Q(0, Z)
            for sx in range(0, w, 10):
                cv2.line(sub, (sx, a[1]), (sx + 5, a[1]), (58, 68, 80), 1)
        canvas[roi] = sub
        for Z in (c["near_m"], c["mid_m"], R):
            self.text(x + w - 6, P(0, Z)[1] - 2, f"{Z:.0f} m", 12, DIM, False, "rb")
        self.text(x + 8, y + 6, "TOP VIEW", 11, DIM, True)

        # ego vehicle
        e0, e1 = P(-0.9, 0), P(0.9, -4.5)
        cv2.rectangle(canvas, (e0[0], e0[1]), (e1[0], e1[1]), (196, 204, 212), -1, cv2.LINE_AA)
        cv2.rectangle(canvas, P(-0.75, -1.2), P(0.75, -2.0), (40, 48, 58), -1)

        size = {"person": (0.6, 0.6), "bicycle": (0.7, 1.8), "motorcycle": (0.8, 2.0), "car": (1.8, 4.5), "truck": (2.5, 8.0),
                "bus": (2.6, 11.0), "train": (3.0, 15.0), "rider": (0.7, 1.8), "cat": (0.4, 0.6), "dog": (0.5, 0.9)}
        for o in sorted(dec.obstacles, key=lambda o: o is alert.obj):
            d = o.corridor_distance if o.corridor_distance is not None else o.distance
            if d > R:
                continue
            ow, ol = size.get(o.name, (1.0, 1.0))
            ow = max(ow, o.x_hi - o.x_lo) if o.name not in ("person", "cat", "dog") else ow
            focus = o is alert.obj
            col = ALERT_MARK[alert.state] if focus else ((150, 160, 170) if o.corridor_distance is not None else (88, 98, 110))
            p0, p1 = P(o.x_m - ow / 2, d + min(ol, R - d)), P(o.x_m + ow / 2, d)
            if o.name == "person":
                cv2.circle(canvas, P(o.x_m, d + 0.3), max(5, int(0.4 * xpm)), col, -1, cv2.LINE_AA)
            else:
                cv2.rectangle(canvas, p0, p1, col, -1, cv2.LINE_AA)
            if focus:
                cv2.rectangle(canvas, (p0[0] - 3, p0[1] - 3), (p1[0] + 3, p1[1] + 3), (255, 255, 255), 1, cv2.LINE_AA)
                label = f"{o.name} {d:.1f} m"
                lx = min(max(p0[0], x + 4), x + w - 6 - int(font(13, True).getlength(label)))
                self.text(lx, max(p0[1] - 6, y + 34), label, 13, TEXT, True, "lb", bg=(11, 14, 18))

    def _timeline(self, canvas, rect) -> None:
        x, y, w, h = rect
        self.text(x, y, f"ALERT HISTORY  (last {self.history.maxlen} updates, newest right)", 11, DIM)
        by, n = y + 16, self.history.maxlen
        cw = (w - (n - 1)) / n
        for i in range(n):
            j = i - (n - len(self.history))
            st = self.history[j] if j >= 0 else None
            col = ALERT_MARK[st] if st in ("WARNING", "CAUTION") else ((52, 60, 70) if st else (24, 30, 37))
            x0 = int(x + i * (cw + 1))
            cv2.rectangle(canvas, (x0, by), (int(x0 + cw), by + h - 18), col, -1)

    def _hud(self, canvas, i: int, dec: Decision, cam: Camera, alert, info: Info) -> None:
        x, y, w, h = self._panel_frame(canvas, i, "DRIVER ALERT", "forward collision warning")
        pad = 12
        dev = self.view == "research"
        if w > 1.4 * h:  # wide panel (research view): card + timeline left, top view right, dev lines below
            body_h = h - (44 if dev else 0) - pad
            lw = int(w * 0.54)
            self._alert_card(canvas, (x + pad, y + pad, lw - pad, body_h - 52), alert)
            self._timeline(canvas, (x + pad, y + pad + body_h - 42, lw - pad, 40))
            self._bev(canvas, (x + lw + pad, y + pad, w - lw - 2 * pad, body_h), dec, cam, alert)
        else:  # tall panel (driver view): card, top view, timeline
            card_h = int(h * 0.36)
            self._alert_card(canvas, (x + pad, y + pad, w - 2 * pad, card_h), alert)
            self._bev(canvas, (x + pad, y + 2 * pad + card_h, w - 2 * pad, h - card_h - 3 * pad - 46), dec, cam, alert)
            self._timeline(canvas, (x + pad, y + h - pad - 40, w - 2 * pad, 40))
        if dev:
            t = info.timings
            self.text(x + pad, y + h - 24, f"DA3 {t.get('depth', 0):.0f} ms  SegFormer {t.get('seg', 0):.0f} ms  YOLO {t.get('det', 0):.0f} ms  "
                                           f"fusion {t.get('fusion', 0):.0f} ms  render {t.get('render', 0):.0f} ms  {info.fps:.1f} FPS  |  "
                                           f"zone {dec.level}  field score {dec.score:.2f}", 11, DIM, False, "lb")
            self.text(x + pad, y + h - 7, info.ref_text, 11, info.ref_color, False, "lb")

    def _header(self, canvas, info: Info):
        cv2.rectangle(canvas, (0, 0), (self.w, HEADER_H), (6, 8, 11), -1)
        cv2.line(canvas, (0, HEADER_H - 1), (self.w, HEADER_H - 1), BORDER, 1)
        cv2.circle(canvas, (16, HEADER_H // 2), 6, (236, 56, 56) if info.live else (240, 168, 40), -1, cv2.LINE_AA)
        self.text(28, HEADER_H // 2, "LIVE" if info.live else "REPLAY", 13, TEXT, True, "lm")
        self.text(100, HEADER_H // 2, "ROAD RISK ASSESSMENT", 17, TEXT, True, "lm")
        self.text(330, HEADER_H // 2, f"{self.view} view  |  camera-only  |  DA3 metric depth + YOLO26-seg + SegFormer-B2", 12, DIM, False, "lm")
        total = f"/{info.total}" if info.total else ""
        self.text(self.w - 14, HEADER_H // 2, f"{info.source}    frame {info.frame}{total}    {info.fps:.1f} FPS    {info.gpu}",
                  13, TEXT, False, "rm")

    def render(self, rgb: np.ndarray, class_map: np.ndarray, depth: np.ndarray, dec: Decision, cam: Camera, info: Info, alert) -> np.ndarray:
        """RGB canvas of the full dashboard."""
        self.history.append(alert.state)
        canvas = np.empty((self.h, self.w, 3), np.uint8)
        canvas[:] = BG
        self._header(canvas, info)
        self._camera(canvas, rgb, dec, cam, depth.shape, info, alert)
        if self.view == "driver":
            self._hud(canvas, 1, dec, cam, alert, info)
        else:
            self._segmentation(canvas, rgb, class_map)
            self._depth(canvas, depth, dec, alert)
            self._hud(canvas, 3, dec, cam, alert, info)
        return self._flush_text(canvas)
