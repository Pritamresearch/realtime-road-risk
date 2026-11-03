"""Offline evaluation of fusion variants on the cached 242 annotated frames (run eval_cache.py first).

    python evaluate.py                 # grid from eval_grid.yaml -> out/eval_results.md, out/eval_frames.csv

Every variant goes through risk.assess with the live camera, ego-mask and segmentation-confidence settings from config.yaml.
Selection and reporting follow eval_grid.yaml (pre-declared): LOCO on per-clip-mean top-1, two-stage bootstrap intervals.
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from models import IGNORE
from risk import assess, camera_for, instance_candidates, seg_candidates

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache" / "eval"


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def auroc(score: np.ndarray, pos: np.ndarray) -> float:
    """Probability a critical frame outscores a nothing-critical frame (ties count half)."""
    p, n = score[pos], score[~pos]
    if not len(p) or not len(n):
        return float("nan")
    return float(((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))


def frame_setup(z, clip: str, cfg: dict, res: int):
    """Depth in meters, camera and ego mask for one cached frame, as run.py would build them."""
    raw = z[f"raw{res}"].astype(np.float32)
    portrait = bool(z["portrait"])
    cam_cfg = dict(cfg["camera"]["portrait" if portrait else "default"])
    cam_cfg.update(cfg["camera"].get("per_source", {}).get(clip, {}))
    f_px = (raw.shape[1] / 2) / math.tan(math.radians(cam_cfg["hfov_deg"]) / 2)
    depth = f_px * raw / 300.0
    ego = None
    if cam_cfg.get("ego_mask"):
        ego = np.zeros(raw.shape, np.uint8)
        cv2.fillPoly(ego, [np.round(np.array(p) * (raw.shape[1], raw.shape[0])).astype(np.int32) for p in cam_cfg["ego_mask"]], 1)
        ego = ego.astype(bool)
    return depth, camera_for(raw.shape, f_px, cam_cfg["horizon"], cam_cfg["height_m"]), ego


def on_grid(arr: np.ndarray, shape) -> np.ndarray:
    return arr if arr.shape == tuple(shape) else cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def candidates_for(z, source: str, weights: dict, conf: float, shape, ego, cfg: dict):
    if source == "seg":
        seg = on_grid(z["seg"], shape).copy()
        seg[on_grid(z["seg_prob"], shape) < cfg["risk"]["min_seg_conf"] * 255] = IGNORE
        if ego is not None:
            seg[ego] = IGNORE
        return seg_candidates(seg, weights, cfg["risk"]["min_object_px"])
    label = on_grid(z["yolo_label"], shape).copy()
    if ego is not None:
        label[ego] = 0
    return instance_candidates(label, z["yolo_cls"], z["yolo_conf"], z["yolo_box"], weights, conf, cfg["risk"]["min_object_px"])


def hit_of(box, shape, refs) -> tuple[float, float]:
    if box is None or not len(refs):
        return 0.0, 0.0
    h, w = shape
    b = (box[0] / w, box[1] / h, box[2] / w, box[3] / h)
    best = max(iou(b, r) for r in refs)
    return float(best >= 0.5), best


def score_frames(cfg: dict, grid: dict, res: int = 504) -> pd.DataFrame:
    """One row per (frame, variant): hit (boxed frames), frame score, level."""
    rows = []
    g = grid["grid"]
    base_keys = [(s, c, k, q) for s in g["source"] for c in (g["yolo_conf"] if s == "yolo" else [None])
                 for k in g["classes"] for q in g["dist_q"]]
    files = sorted(CACHE.glob("*/*.npz"))
    for i, p in enumerate(files):
        clip, frame = p.parent.name, int(p.stem)
        z = np.load(p)
        refs = z["ref_box"]
        depth, cam, ego = frame_setup(z, clip, cfg, res)
        meta = {"clip": clip, "frame": frame, "boxed": len(refs) > 0}

        def add(name, box, score, level):
            hit, best = hit_of(box, depth.shape, refs)
            rows.append({**meta, "variant": name, "hit": hit, "iou": best, "score": score, "level": level})

        for src, conf, cls_set, dq in base_keys:
            weights = grid["class_sets"][cls_set]
            rcfg = {**cfg["risk"], "class_weights": weights, "dist_q": dq, "rule": "corridor",
                    "field_lambda_m": 20.0, "field_sigma_m": math.inf}
            dec = assess(candidates_for(z, src, weights, conf or 0.0, depth.shape, ego, cfg), depth, cam, rcfg)
            stem = f"{src}|{conf or '-'}|{cls_set}|q{dq}"
            add(f"{stem}|corridor", dec.top.box if dec.top else None, dec.score, dec.level)
            w_max = max(weights.values())
            for lam, sig in itertools.product(g["field_lambda_m"], g["field_sigma_m"]):  # same objects, field rule
                best, best_key = None, None
                for o in dec.obstacles:
                    lat = 1.0 if math.isinf(sig) else math.exp(-o.gap_m ** 2 / (2 * sig * sig))
                    f = o.weight / w_max * math.exp(-o.distance / lam) * lat
                    if best_key is None or (f, o.area) > best_key:
                        best, best_key = o, (f, o.area)
                add(f"{stem}|field|l{lam}|s{sig}", best.box if best else None, best_key[0] if best else 0.0, dec.level)

        if res == 504:  # fixed references
            weights = grid["class_sets"]["paper"]
            label = z["yolo_label"].copy()
            if ego is not None:
                label[on_grid(ego.astype(np.uint8), label.shape).astype(bool)] = 0
            objs = instance_candidates(label, z["yolo_cls"], z["yolo_conf"], z["yolo_box"], weights, 0.25, 1)
            h = depth.shape[0]
            for name, key in (("v1 box height", lambda o: o.weight * (o.box[3] - o.box[1]) / h),
                              ("v1 box bottom", lambda o: o.weight * o.box[3] / h), ("v1 semantic only", lambda o: o.weight)):
                best = max(objs, key=lambda o: (key(o), o.mask.sum())) if objs else None
                add(name, best.box if best else None, key(best) if best else 0.0, "")
            old = {"person": 1, "rider": 1, "car": 1, "truck": 1, "bus": 1, "train": 1, "motorcycle": 1, "bicycle": 1}
            rcfg = {**cfg["risk"], "class_weights": old, "dist_q": 5, "rule": "corridor", "field_lambda_m": 20.0, "field_sigma_m": math.inf}
            dec = assess(candidates_for(z, "seg", old, 0.0, depth.shape, ego, cfg), depth, cam, rcfg)
            add("previous realtime (seg, corridor)", dec.top.box if dec.top else None, dec.score, dec.level)
        if (i + 1) % 40 == 0:
            print(f"scored {i + 1}/{len(files)} frames", flush=True)
    return pd.DataFrame(rows)


def per_clip_rate(df: pd.DataFrame) -> pd.DataFrame:
    """variant x clip mean top-1 over boxed frames."""
    return df[df.boxed].pivot_table(index="variant", columns="clip", values="hit", aggfunc="mean")


def loco(df: pd.DataFrame, variants: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    rate = per_clip_rate(df[df.variant.isin(variants)]).reindex(variants)
    choice, parts = {}, []
    for c in rate.columns:
        best = variants[int(np.argmax(rate.drop(columns=c).mean(axis=1).to_numpy()))]  # first maximum = grid order
        choice[c] = best
        parts.append(df[(df.variant == best) & (df["clip"] == c)])
    return pd.concat(parts).assign(variant="LOCO-selected"), choice


def bootstrap_draws(df: pd.DataFrame, n: int, seed: int) -> list[np.ndarray]:
    """Two-stage bootstrap over the boxed frames of one variant's frame order: clips with replacement, then frames."""
    rng = np.random.default_rng(seed)
    keys = df[df.boxed][["clip", "frame"]].drop_duplicates().reset_index(drop=True)
    by_clip = {c: g.index.to_numpy() for c, g in keys.groupby("clip")}
    clips = list(by_clip)
    draws = []
    for _ in range(n):
        idx = [rng.choice(by_clip[c], size=len(by_clip[c])) for c in rng.choice(clips, size=len(clips))]
        draws.append(np.concatenate(idx))
    return draws, keys


def ci(values: np.ndarray, draws) -> tuple[float, float, float]:
    b = np.array([values[d].mean() for d in draws])
    return float(values.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> None:
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    grid = yaml.safe_load((HERE / "eval_grid.yaml").read_text())
    df = score_frames(cfg, grid)
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "eval_frames.csv", index=False)

    refs = ["v1 box height", "v1 box bottom", "v1 semantic only", "previous realtime (seg, corridor)"]
    cands = [v for v in df.variant.unique() if v not in refs]
    sel, choice = loco(df, cands)
    rate = per_clip_rate(df[df.variant.isin(cands)]).reindex(cands)
    full_best = cands[int(np.argmax(rate.mean(axis=1).to_numpy()))]

    draws, keys = bootstrap_draws(df[df.variant == refs[0]], grid["n_boot"], grid["seed"])

    def aligned(frame_df: pd.DataFrame) -> np.ndarray:
        m = keys.merge(frame_df[frame_df.boxed][["clip", "frame", "hit"]], on=["clip", "frame"], how="left")
        return m["hit"].to_numpy(float)

    table = []
    ref_hits = aligned(df[df.variant == "v1 box height"])
    for name, fdf in [("LOCO-selected (held out)", sel)] + [(r, df[df.variant == r]) for r in refs] + \
                     [(f"best on all clips (in-sample): {full_best}", df[df.variant == full_best])]:
        hits = aligned(fdf)
        m, lo, hi = ci(hits, draws)
        d, dlo, dhi = ci(hits - ref_hits, draws)
        a = auroc(fdf["score"].to_numpy(float), fdf["boxed"].to_numpy(bool))
        table.append({"method": name, "top1": m, "lo": lo, "hi": hi, "minus_box_height": d, "d_lo": dlo, "d_hi": dhi, "auroc": a})
    t = pd.DataFrame(table)

    # family view: best config per (source, rule) by in-sample per-clip mean (descriptive only)
    fam = rate.mean(axis=1).rename("macro").to_frame()
    fam["family"] = [("yolo" if v.startswith("yolo") else "seg") + ("|field" if "|field|" in v else "|corridor") for v in fam.index]
    fam_best = fam.sort_values("macro", ascending=False).groupby("family").head(1)
    oracle = df[df.boxed & (df.variant.isin(cands))].groupby(["clip", "frame"]).hit.max().mean()

    lines = ["# Offline evaluation on the 242 annotated own-clip frames", "",
             f"{df[['clip', 'frame']].drop_duplicates().shape[0]} frames, {keys.shape[0]} with boxes, {len(cands)} grid variants "
             "(eval_grid.yaml, declared before scoring). Top-1 hit: the selected object's box has IoU >= 0.5 with a human box; "
             "pooled over boxed frames. Intervals: two-stage bootstrap (clips, then frames), 2,000 draws, seed 0; LOCO selection is "
             "not re-run inside the bootstrap. AUROC: frame score, boxed vs nothing-critical frames (pooled; for LOCO the held-out "
             "clips' scores come from different configs).", "",
             "| Method | top-1 hit [95% CI] | minus box height [95% CI] | frame AUROC |", "|---|---|---|---|"]
    for r in t.itertuples():
        lines.append(f"| {r.method} | {r.top1:.3f} [{r.lo:.3f}, {r.hi:.3f}] | {r.minus_box_height:+.3f} [{r.d_lo:+.3f}, {r.d_hi:+.3f}] | {r.auroc:.3f} |")
    lines += ["", f"Oracle ceiling (any grid variant hits): {oracle:.3f}", "", "LOCO choice per held-out clip:", ""]
    lines += [f"- {c}: `{v}`" for c, v in choice.items()]
    lines += ["", "Best config per family (in-sample per-clip mean, descriptive):", ""]
    lines += [f"- {r.family}: `{r.Index}` {r.macro:.3f}" for r in fam_best.itertuples()]
    df392 = score_frames(cfg, grid, 392)
    df392.to_csv(out / "eval_frames_392.csv", index=False)
    sel392 = pd.concat([df392[(df392.variant == v) & (df392["clip"] == c)] for c, v in choice.items()])
    lines += ["", "Depth at 392 px (--fast), same choices:", ""]
    for name, a, b in (("LOCO-selected (held out)", sel, sel392), (full_best, df[df.variant == full_best], df392[df392.variant == full_best])):
        lines.append(f"- {name}: top-1 {aligned(a).mean():.3f} at 504 px -> {aligned(b).mean():.3f} at 392 px; "
                     f"AUROC {auroc(a.score.to_numpy(float), a.boxed.to_numpy(bool)):.3f} -> {auroc(b.score.to_numpy(float), b.boxed.to_numpy(bool)):.3f}")
    (out / "eval_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
