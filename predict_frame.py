"""Run the pipeline on single frames: dashboard PNG + decision JSON per image.

    python predict_frame.py frame.jpg                         # -> out/frame_dashboard.png, out/frame.json
    python predict_frame.py a.jpg b.png --out results/        # several frames, one model load
    python predict_frame.py frame.jpg --hfov-deg 70 --horizon 0.48 --height-m 1.4    # camera assumptions
    python predict_frame.py drive.mp4 --frame 120             # one frame (1-based) taken from a video
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def read_video_frame(path: Path, number: int):
    """Frame `number` (1-based) of a video, rotation metadata applied (sequential read: seeking is not frame-exact)."""
    from sources import VideoSource

    for frame in VideoSource(path).frames():
        if frame.index == number:
            return frame.bgr
    raise ValueError(f"{path} has fewer than {number} frames")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="image files, or one video with --frame")
    ap.add_argument("--frame", type=int, help="take this 1-based frame from a video input")
    ap.add_argument("--out", default="out")
    ap.add_argument("--hfov-deg", dest="hfov_deg", type=float)
    ap.add_argument("--horizon", type=float)
    ap.add_argument("--height-m", dest="height_m", type=float)
    ap.add_argument("--hq", action="store_true")
    args = ap.parse_args()

    from pipeline import RiskPipeline

    pipe = RiskPipeline(hq=args.hq, camera={"hfov_deg": args.hfov_deg, "horizon": args.horizon, "height_m": args.height_m})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.inputs:
        path = Path(name)
        bgr = read_video_frame(path, args.frame) if args.frame else cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(path)
        stem = path.stem + (f"_f{args.frame:06d}" if args.frame else "")
        result = pipe(bgr, path.stem)
        cv2.imwrite(str(out / f"{stem}_dashboard.png"), cv2.cvtColor(pipe.render(result, name=stem), cv2.COLOR_RGB2BGR))
        summary = result.to_dict()
        (out / f"{stem}.json").write_text(json.dumps(summary, indent=2))
        top = summary["most_critical"]
        print(f"{name}: {summary['level']}"
              + (f" at {summary['level_distance_m']} m" if summary["level_distance_m"] is not None else "")
              + (f", most critical: {top['class']} {top['distance_m']} m" if top else "")
              + f"  -> {out / (stem + '_dashboard.png')}")


if __name__ == "__main__":
    main()
