"""Four-panel dashboard: camera with corridor and obstacle distances, segmentation, metric depth, risk decision.

Shapes are drawn with OpenCV on an RGB canvas; text is queued and drawn in one PIL pass at the end (monospace TrueType).
"""
from __future__ import annotations

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
LEVEL_RGB = {"NEAR": (236, 72, 64), "MID": (240, 168, 40), "FAR": (60, 200, 110), "CLEAR": (120, 134, 146)}
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
    def __init__(self, width: int, height: int, risk_cfg: dict, history: int = 60):
        self.w, self.h, self.cfg = width, height, risk_cfg
        self.history: deque[tuple[float, str]] = deque(maxlen=history)
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
    def _camera(self, canvas, rgb, dec: Decision, cam: Camera, depth_hw, info: Info):
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

        # objects; the most critical one (selection rule) is highlighted in the colour of its distance zone
        for o in sorted(dec.obstacles, key=lambda o: -o.distance):
            p0, p1 = to_disp(o.box[0], o.box[1]), to_disp(o.box[2], o.box[3])
            is_top = dec.top is o
            if is_top:
                col = LEVEL_RGB[self.zone_of(object_distance(o))]
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

        badge = f"{dec.level}  {dec.distance:.1f} m" if dec.distance is not None else dec.level
        self.text(rect[0] + 10, rect[1] + rect[3] - 10, badge, 16, (10, 14, 18), True, "lb", bg=LEVEL_RGB[dec.level])

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

    def _depth(self, canvas, depth, dec: Decision):
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

        n = dec.top
        if n is not None:
            px = int(fx + (n.point[0] + 0.5) * fw / depth.shape[1])
            py = int(fy + (n.point[1] + 0.5) * fh / depth.shape[0])
            cv2.circle(canvas, (px, py), 7, ACCENT, 2, cv2.LINE_AA)
            cv2.line(canvas, (px - 11, py), (px + 11, py), ACCENT, 1, cv2.LINE_AA)
            cv2.line(canvas, (px, py - 11), (px, py + 11), ACCENT, 1, cv2.LINE_AA)
            label = f"d= {object_distance(n):.1f} m"
            lx = min(px + 12, fx + fw - 4 - int(font(12, True).getlength(label)))
            self.text(lx, max(py - 10, fy + 18), label, 12, (8, 20, 26), True, "lb", bg=ACCENT)

    def _decision(self, canvas, dec: Decision, info: Info):
        x, y, w, h = self._panel_frame(canvas, 3, "RISK DECISION", "late fusion")
        c = self.cfg
        col = LEVEL_RGB[dec.level]
        # level box
        bx, by, bw, bh = x + 14, y + 12, 236, 118
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), tuple(int(v * 0.16) for v in col), -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), col, 2)
        self.text(bx + 10, by + 10, "RISK LEVEL", 11, DIM)
        self.text(bx + bw // 2, by + bh // 2 + 4, dec.level, 46, col, True, "mm")
        ranges = {"NEAR": f"< {c['near_m']:.1f} m", "MID": f"{c['near_m']:.0f}-{c['mid_m']:.0f} m",
                  "FAR": f"> {c['mid_m']:.1f} m", "CLEAR": "no obstacle in corridor"}
        self.text(bx + 10, by + bh - 8, ranges[dec.level], 11, DIM, False, "lb")

        # most critical object + score
        rx, rw = bx + bw + 22, w - bw - 50
        self.text(rx, by + 4, "MOST CRITICAL OBJECT" + ("  (in corridor)" if dec.top is not None and dec.top.corridor_distance is not None else ""), 11, DIM)
        if dec.top is not None:
            self.text(rx, by + 24, dec.top.name, 22, TEXT, True)
            self.text(rx + 20 + int(font(22, True).getlength(dec.top.name)), by + 24, f"{object_distance(dec.top):.1f} m", 22, TEXT, True)
        else:
            self.text(rx, by + 24, "--", 22, DIM, True)
        self.text(rx, by + 66, "RISK SCORE", 11, DIM)
        self.text(rx + rw, by + 66, f"{dec.score:.2f}", 13, TEXT, True, "ra")
        cv2.rectangle(canvas, (rx, by + 86), (rx + rw, by + 98), (30, 38, 46), -1)
        cv2.rectangle(canvas, (rx, by + 86), (rx + int(rw * dec.score), by + 98), col, -1)
        cv2.rectangle(canvas, (rx, by + 86), (rx + rw, by + 98), BORDER, 1)

        # zone bars
        zy = by + bh + 18
        self.text(x + 14, zy, "CORRIDOR OBSTACLE PIXELS BY ZONE", 11, DIM)
        total = max(sum(z.px for z in dec.zones), 1)
        bar_x0, bar_x1 = x + 140, x + w - 260
        for i, z in enumerate(dec.zones):
            ry = zy + 26 + i * 22
            label = {"NEAR": f"NEAR  <{z.hi:.0f} m", "MID": f"MID   {z.lo:.0f}-{z.hi:.0f} m", "FAR": f"FAR   >{z.lo:.0f} m"}[z.name]
            self.text(x + 14, ry, label, 12, LEVEL_RGB[z.name], False, "lm")
            frac = z.px / total if dec.corridor_px else 0.0
            cv2.rectangle(canvas, (bar_x0, ry - 6), (bar_x1, ry + 6), (30, 38, 46), -1)
            if frac > 0:
                cv2.rectangle(canvas, (bar_x0, ry - 6), (bar_x0 + int((bar_x1 - bar_x0) * frac), ry + 6), LEVEL_RGB[z.name], -1)
            self.text(bar_x1 + 52, ry, f"{100 * frac:.1f}%", 12, TEXT, False, "rm")
            self.text(bar_x1 + 120, ry, f"{z.px} px", 12, TEXT, False, "rm")
            self.text(bar_x1 + 134, ry, f"min {z.min_m:.1f} m" if z.min_m is not None else "min --", 12, TEXT if z.min_m else DIM, False, "lm")

        # history
        hy = zy + 26 + 3 * 22
        self.text(x + 14, hy, f"RISK SCORE HISTORY (last {self.history.maxlen} frames)", 11, DIM)
        gx0, gy0, gx1, gy1 = x + 14, hy + 18, x + w - 14, y + h - 66
        cv2.rectangle(canvas, (gx0, gy0), (gx1, gy1), (17, 22, 28), -1)
        cv2.rectangle(canvas, (gx0, gy0), (gx1, gy1), BORDER, 1)
        for thr, name in ((c["near_m"], "NEAR"), (c["mid_m"], "MID")) if c["rule"] == "corridor" else ():
            s = np.clip((c["mid_m"] - thr) / (c["mid_m"] - c["critical_m"]), 0, 1)
            ty = int(gy1 - s * (gy1 - gy0 - 4)) - 2
            for sx in range(gx0 + 2, gx1 - 2, 8):
                cv2.line(canvas, (sx, ty), (min(sx + 3, gx1 - 2), ty), tuple(v // 3 for v in LEVEL_RGB[name]), 1)
        n = self.history.maxlen
        pts = [(int(gx1 - 4 - (len(self.history) - 1 - i) * (gx1 - gx0 - 8) / max(n - 1, 1)), int(gy1 - 2 - s * (gy1 - gy0 - 4)), lv)
               for i, (s, lv) in enumerate(self.history)]
        for (ax, ay, _), (cx_, cy_, lv) in zip(pts, pts[1:]):
            cv2.line(canvas, (ax, ay), (cx_, cy_), (240, 168, 40), 1, cv2.LINE_AA)
        for px, py, lv in pts:
            cv2.circle(canvas, (px, py), 3, LEVEL_RGB[lv], -1, cv2.LINE_AA)

        t = info.timings
        self.text(x + 14, y + h - 44, f"DA3 {t.get('depth', 0):.0f} ms   SegFormer {t.get('seg', 0):.0f} ms   fusion {t.get('fusion', 0):.0f} ms   "
                                      f"render {t.get('render', 0):.0f} ms   pipeline {info.fps:.1f} FPS", 12, TEXT, False, "lb")
        self.text(x + 14, y + h - 26, info.params + f"   corridor px={dec.corridor_px}", 12, DIM, False, "lb")
        self.text(x + 14, y + h - 8, info.ref_text, 12, info.ref_color, False, "lb")

    def _header(self, canvas, info: Info):
        cv2.rectangle(canvas, (0, 0), (self.w, HEADER_H), (6, 8, 11), -1)
        cv2.line(canvas, (0, HEADER_H - 1), (self.w, HEADER_H - 1), BORDER, 1)
        cv2.circle(canvas, (16, HEADER_H // 2), 6, (236, 56, 56) if info.live else (240, 168, 40), -1, cv2.LINE_AA)
        self.text(28, HEADER_H // 2, "LIVE" if info.live else "REPLAY", 13, TEXT, True, "lm")
        self.text(100, HEADER_H // 2, "ROAD RISK ASSESSMENT", 17, TEXT, True, "lm")
        self.text(330, HEADER_H // 2, "camera-only  |  SegFormer-B2 + DA3 metric depth  |  monocular", 12, DIM, False, "lm")
        total = f"/{info.total}" if info.total else ""
        self.text(self.w - 14, HEADER_H // 2, f"{info.source}    frame {info.frame}{total}    {info.fps:.1f} FPS    {info.gpu}",
                  13, TEXT, False, "rm")

    def render(self, rgb: np.ndarray, class_map: np.ndarray, depth: np.ndarray, dec: Decision, cam: Camera, info: Info) -> np.ndarray:
        """RGB canvas of the full dashboard."""
        self.history.append((dec.score, dec.level))
        canvas = np.empty((self.h, self.w, 3), np.uint8)
        canvas[:] = BG
        self._header(canvas, info)
        self._camera(canvas, rgb, dec, cam, depth.shape, info)
        self._segmentation(canvas, rgb, class_map)
        self._depth(canvas, depth, dec)
        self._decision(canvas, dec, info)
        return self._flush_text(canvas)
