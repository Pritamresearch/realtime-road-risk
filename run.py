# Copyright (C) 2026 Pritam Chakraborty <pritam.23.research@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Real-time road risk assessment from one camera: DA3 metric depth + YOLO26-seg + SegFormer-B2, late fusion, live dashboard.

Examples (from this folder):
    python run.py path/to/drive.mp4                   # a video, real-time playback (frames dropped to keep up)
    python run.py 0                                   # webcam 0
    python run.py path/to/frame.jpg                   # a single image (or a folder of images)
    python run.py drive.mp4 --mode all --save out/drive.mp4 --log out/drive.csv   # every frame, recorded
    python run.py sikim --annotated                   # own clips only: the human-annotated frames, with MATCH/MISS
Keys: q / Esc quit, space pause, s save a screenshot to out/.
"""
from __future__ import annotations

import argparse
import csv
import queue
import threading
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def compare_reference(dec, ref, depth_hw) -> tuple[str, tuple[int, int, int], dict]:
    """Eval-only line for the dashboard and the log. Annotator frames: top-1 hit = the most critical object's box has IoU >= 0.5
    with a human box (object-level top-1 criterion). BDD-OIA: the action label is shown, not scored."""
    from dashboard import DIM, LEVEL_RGB

    if ref is None:
        return "REF (eval only): none for this frame", DIM, {}
    if ref.kind == "bdd_oia":
        return f"REF (eval only): BDD-OIA action = {ref.label}", (190, 200, 210), {"ref_label": ref.label}
    if not ref.boxes:
        return "REF (eval only): annotator marked nothing safety-critical", (190, 200, 210), {"ref_label": "none"}
    h, w = depth_hw
    iou = 0.0
    if dec.top is not None:
        b = dec.top.box
        iou = max(box_iou((b[0] / w, b[1] / h, b[2] / w, b[3] / h), r) for r in ref.boxes)
    hit = iou >= 0.5
    text = f"REF (eval only): annotator {len(ref.boxes)} box{'es' if len(ref.boxes) > 1 else ''}, top-1 IoU {iou:.2f}  [{'MATCH' if hit else 'MISS'}]"
    return text, LEVEL_RGB["FAR"] if hit else LEVEL_RGB["NEAR"], {"ref_label": f"{len(ref.boxes)} boxes", "ref_iou": round(iou, 3), "ref_hit": int(hit)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="video, image, image folder, camera index, bdd_oia[:split], or part of a file name in data/videos")
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--mode", choices=["live", "all"], help="live: real-time playback, drop frames; all: process every frame (default: live for videos/cameras)")
    ap.add_argument("--every", type=int, default=1, help="mode all: process every n-th frame")
    ap.add_argument("--annotated", action="store_true", help="process only the frames of an own clip that carry human annotations")
    ap.add_argument("--n", type=int, default=200, help="bdd_oia: number of frames (seeded sample)")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--hfov-deg", dest="hfov_deg", type=float, help="override the assumed horizontal field of view")
    ap.add_argument("--horizon", type=float, help="override the horizon row (fraction of image height)")
    ap.add_argument("--height-m", dest="height_m", type=float, help="override the camera height above the road")
    ap.add_argument("--save", help="write the dashboard to this .mp4")
    ap.add_argument("--log", help="write per-frame decisions to this .csv")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--hq", action="store_true", help="larger model inputs (models.hq in the config), ~1.5x slower")
    ap.add_argument("--view", choices=["research", "driver"], default="research",
                    help="research: camera, segmentation, depth, driver alert; driver: camera + driver alert only")
    args = ap.parse_args()

    from dashboard import Info
    from models import gpu_name
    from pipeline import RiskPipeline
    from sources import LatestFrame, open_source

    src = open_source(args.source, args.annotated, args.every, args.n)
    mode = args.mode or ("all" if args.annotated or src.total is not None and src.fps <= 2.0 else "live")
    live = mode == "live"

    print(f"loading models on {gpu_name()} ...", flush=True)
    pipe = RiskPipeline(args.config, args.hq, {"hfov_deg": args.hfov_deg, "horizon": args.horizon, "height_m": args.height_m}, args.view)
    gpu = gpu_name()

    viewer = None
    if not args.no_window:
        from viewer import Viewer

        viewer = Viewer("Road Risk Assessment")
    writer = None
    log_file = log = None
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        log_file = open(args.log, "w", newline="")
        log = csv.DictWriter(log_file, ["frame", "alert", "alert_object", "ttc_s", "closing_mps", "level", "distance_m", "top", "top_distance_m", "score", "near_px", "mid_px", "far_px",
                                        "corridor_px", "hfov_deg", "t_depth_ms", "t_seg_ms", "t_det_ms", "t_fusion_ms",
                                        "ref_label", "ref_iou", "ref_hit"])
        log.writeheader()

    # Two stages: a GPU thread runs the models on frame t+1 while the main thread fuses, draws and shows frame t.
    reader = LatestFrame(src) if live else None
    results: queue.Queue = queue.Queue(maxsize=1)
    stop = threading.Event()

    def perceive() -> None:
        try:
            for frame in (iter(reader) if live else src.frames()):
                if stop.is_set():
                    break
                item = (frame, pipe.perceive(frame.bgr, src.name, frame.t))
                while not stop.is_set():
                    try:
                        results.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except Exception as e:  # surfaced on the main thread
            results.put(e)
        results.put(None)

    threading.Thread(target=perceive, daemon=True).start()
    fps_ema, render_ms, levels, hits = 0.0, 0.0, Counter(), []
    t_prev = time.perf_counter()
    count = 0
    try:
        while True:
            while True:  # keep the window responsive while the GPU stage works
                try:
                    item = results.get(timeout=0.05)
                    break
                except queue.Empty:
                    if viewer:
                        viewer.poll()
                        if viewer.closed:
                            raise KeyboardInterrupt
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            frame, perception = item
            count += 1
            res = pipe.decide(perception)
            dec, timings = res.decision, perception.timings
            timings["render"] = render_ms
            t4 = time.perf_counter()

            ref_text, ref_color, ref_log = compare_reference(dec, frame.ref, perception.depth.shape)
            info = Info(src.name, frame.index, src.total, live, fps_ema, gpu, timings, pipe.params_text(res), ref_text, ref_color,
                        frame.ref.boxes if frame.ref is not None else [], perception.camera.get("ego_mask", []))
            canvas = cv2.cvtColor(pipe.render(res, info), cv2.COLOR_RGB2BGR)
            t5 = time.perf_counter()
            render_ms = 1e3 * (t5 - t4)
            dt = t5 - t_prev
            t_prev = t5
            fps_ema = 1.0 / dt if fps_ema == 0 else 0.8 * fps_ema + 0.2 / dt

            levels[res.alert.state] += 1
            if "ref_hit" in ref_log:
                hits.append(ref_log["ref_hit"])
            if log:
                z = {zz.name: zz.px for zz in dec.zones}
                top_d = None if dec.top is None else (dec.top.corridor_distance if dec.top.corridor_distance is not None else dec.top.distance)
                al = res.alert
                log.writerow({"frame": frame.index, "alert": al.state, "alert_object": al.obj.name if al.obj else "",
                              "ttc_s": None if al.trend.ttc_s is None else round(al.trend.ttc_s, 2),
                              "closing_mps": None if al.trend.closing_mps is None else round(al.trend.closing_mps, 2), "level": dec.level, "distance_m": None if dec.distance is None else round(dec.distance, 2),
                              "top": dec.top.name if dec.top else "", "top_distance_m": None if top_d is None else round(top_d, 2),
                              "score": round(dec.score, 3), "near_px": z["NEAR"], "mid_px": z["MID"], "far_px": z["FAR"],
                              "corridor_px": dec.corridor_px, "hfov_deg": perception.camera["hfov_deg"],
                              "t_depth_ms": round(timings["depth"], 1), "t_seg_ms": round(timings["seg"], 1),
                              "t_det_ms": round(timings["det"], 1), "t_fusion_ms": round(timings["fusion"], 1), **ref_log})
            if args.save:
                if writer is None:
                    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                    out_fps = max(1.0, min(src.fps / args.every, 30.0)) if not live else 5.0
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (canvas.shape[1], canvas.shape[0]))
                writer.write(canvas)
            if viewer:
                keys = viewer.show(canvas[:, :, ::-1])
                if viewer.closed or {"q", "Escape"} & set(keys):
                    break
                if "s" in keys:
                    (HERE / "out").mkdir(exist_ok=True)
                    shot = HERE / "out" / f"{src.name}_{frame.index:06d}.png"
                    cv2.imwrite(str(shot), canvas)
                    print("saved", shot)
                if "space" in keys:
                    if reader:
                        reader.paused = True
                    while not viewer.closed and "space" not in viewer.poll():
                        time.sleep(0.03)
                    if reader:
                        reader.paused = False
                    t_prev = time.perf_counter()
            elif count % 20 == 0:
                print(f"frame {frame.index}  {dec.level:5s}  {'' if dec.distance is None else f'{dec.distance:5.1f} m'}  {fps_ema:.2f} FPS", flush=True)
            if args.max_frames and count >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if writer:
            writer.release()
        if log_file:
            log_file.close()
        if viewer:
            viewer.close()

    n = sum(levels.values())
    print(f"\nprocessed {n} frames, {fps_ema:.2f} FPS at the end; alerts: " + ", ".join(f"{k} {v}" for k, v in levels.most_common()))
    if hits:
        print(f"annotated frames with boxes: {len(hits)}, top-1 hit (IoU >= 0.5): {np.mean(hits):.3f}")


if __name__ == "__main__":
    main()
