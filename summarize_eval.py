"""Summarise `run.py --annotated --log out/eval/<clip>.csv` logs: per-clip top-1 hit on boxed frames and the level mix.

Top-1 hit = the most critical object's box has IoU >= 0.5 with a human "must attend to" box (the v1 object-level criterion).
For the full offline comparison with intervals and LOCO selection use evaluate.py instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "out/eval")
rows = []
for p in sorted(folder.glob("*.csv")):
    df = pd.read_csv(p)
    boxed = df[df["ref_hit"].notna()]
    rows.append({"clip": p.stem, "frames": len(df), "boxed": len(boxed), "hits": int(boxed["ref_hit"].sum()),
                 **{lv: int((df["level"] == lv).sum()) for lv in ("NEAR", "MID", "FAR", "CLEAR")},
                 "ms_depth": round(df["t_depth_ms"].median()), "ms_seg": round(df["t_seg_ms"].median())})
t = pd.DataFrame(rows)
t["top1"] = (t["hits"] / t["boxed"]).round(3)
print(t.to_string(index=False))
print(f"\npooled top-1 hit: {t['hits'].sum()}/{t['boxed'].sum()} = {t['hits'].sum() / t['boxed'].sum():.3f}"
      f"   mean of per-clip: {t['top1'].mean():.3f}")
