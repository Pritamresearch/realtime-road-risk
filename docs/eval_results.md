# Offline evaluation on the 242 annotated own-clip frames

242 frames, 142 with boxes, 120 grid variants (eval_grid.yaml, declared before scoring). Top-1 hit: the selected object's box has IoU >= 0.5 with a human box; pooled over boxed frames. Intervals: two-stage bootstrap (clips, then frames), 2,000 draws, seed 0; LOCO selection is not re-run inside the bootstrap. AUROC: frame score, boxed vs nothing-critical frames (pooled; for LOCO the held-out clips' scores come from different configs).

| Method | top-1 hit [95% CI] | minus box height [95% CI] | frame AUROC |
|---|---|---|---|
| LOCO-selected (held out) | 0.430 [0.250, 0.588] | -0.035 [-0.119, +0.023] | 0.964 |
| v1 box height | 0.465 [0.283, 0.630] | +0.000 [+0.000, +0.000] | 0.936 |
| v1 box bottom | 0.430 [0.241, 0.592] | -0.035 [-0.117, +0.014] | 0.846 |
| v1 semantic only | 0.394 [0.196, 0.636] | -0.070 [-0.158, +0.071] | 0.868 |
| previous realtime (seg, corridor) | 0.218 [0.068, 0.343] | -0.246 [-0.370, -0.146] | 0.871 |
| best on all clips (in-sample): yolo|0.25|road_users|q50|field|l20|s3 | 0.444 [0.271, 0.608] | -0.021 [-0.101, +0.053] | 0.960 |

Oracle ceiling (any grid variant hits): 0.634

LOCO choice per held-out clip:

- 00a0f008-3c67908e: `yolo|0.25|road_users|q50|field|l10|s3`
- bhubaneswar night light: `yolo|0.25|road_users|q50|field|l20|s3`
- challenge_video: `yolo|0.25|road_users|q50|field|l20|s3`
- dhaka: `yolo|0.25|road_users|q50|field|l20|s3`
- dhaka night light: `yolo|0.25|road_users|q50|field|l20|s3`
- india bolpur: `yolo|0.25|road_users|q50|field|l10|sinf`
- malaysia: `yolo|0.25|road_users|q50|field|l20|s3`
- sikim: `yolo|0.25|road_users|q50|field|l20|s3`

Best config per family (in-sample per-clip mean, descriptive):

- yolo|field: `yolo|0.25|road_users|q50|field|l20|s3` 0.551
- seg|field: `seg|-|paper|q10|field|l10|sinf` 0.376
- yolo|corridor: `yolo|0.15|road_users|q10|corridor` 0.312
- seg|corridor: `seg|-|paper|q50|corridor` 0.212

Depth at 392 px (--fast), same choices:

- LOCO-selected (held out): top-1 0.430 at 504 px -> 0.444 at 392 px; AUROC 0.964 -> 0.963
- yolo|0.25|road_users|q50|field|l20|s3: top-1 0.444 at 504 px -> 0.451 at 392 px; AUROC 0.960 -> 0.959
