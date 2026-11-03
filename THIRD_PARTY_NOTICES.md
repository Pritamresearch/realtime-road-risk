# Third-party components

This repository contains only original source code by Pritam Chakraborty (AGPL-3.0-or-later, see `LICENSE`). It does **not**
redistribute model weights, third-party source code or datasets; they are downloaded by the user (`download_weights.py`,
README "Install") and keep their own terms.

## Models (downloaded into `weights/`)

| Component | Source | Licence | Note |
|---|---|---|---|
| Depth Anything 3 Metric-Large (weights) | https://huggingface.co/depth-anything/DA3METRIC-LARGE | Apache-2.0 | |
| YOLO26s-seg (weights) | Ultralytics assets release | AGPL-3.0 | the reason this repository is AGPL-3.0 |
| SegFormer-B2 Cityscapes (weights) | https://huggingface.co/nvidia/segformer-b2-finetuned-cityscapes-1024-1024 | NVIDIA Source Code License for SegFormer | **non-commercial use only**; used for the segmentation panel, not for the risk decision (`risk.source: yolo`) |

## Libraries

| Component | Licence |
|---|---|
| Depth Anything 3 code (installed from source into `third_party/`) | Apache-2.0 |
| ultralytics | AGPL-3.0 |
| PyTorch, torchvision | BSD-3-Clause |
| transformers, huggingface_hub, safetensors | Apache-2.0 |
| OpenCV (opencv-python) | Apache-2.0 |
| NumPy, pandas | BSD-3-Clause |
| Pillow | HPND (MIT-CMU) |
| matplotlib | Matplotlib licence (PSF-based) |
| PyYAML | MIT |

## Data (not included)

The evaluation clips, their annotations and BDD-OIA are not part of the repository. BDD100K and BDD-OIA images are subject to the
BDD100K terms; the public highway video and the author's own recordings are not redistributed. The screenshots in `docs/` show the
author's own recordings only.

**Using the whole system commercially** is restricted by the SegFormer weights (non-commercial), even though the code is AGPL-3.0.
Setting `risk.source: yolo` already keeps SegFormer out of the decision; removing the segmentation panel removes it entirely.
