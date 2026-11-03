"""Download the three model checkpoints into weights/ (about 1.5 GB in total).

    python download_weights.py

DA3METRIC-LARGE (Apache-2.0) and SegFormer-B2 Cityscapes (NVIDIA SegFormer licence, non-commercial) come from Hugging Face;
YOLO26s-seg (AGPL-3.0) is fetched by the ultralytics package.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

WEIGHTS = Path(__file__).resolve().parent / "weights"


def main() -> None:
    WEIGHTS.mkdir(exist_ok=True)
    snapshot_download("depth-anything/DA3METRIC-LARGE", local_dir=WEIGHTS / "da3metric-large")
    print("DA3METRIC-LARGE done")
    snapshot_download("nvidia/segformer-b2-finetuned-cityscapes-1024-1024", local_dir=WEIGHTS / "segformer-b2-cityscapes",
                      allow_patterns=["*.json", "pytorch_model.bin"])  # converted to safetensors on first load (models.py)
    print("SegFormer-B2 done")
    target = WEIGHTS / "yolo26s-seg.pt"
    if not target.exists():
        from ultralytics import YOLO

        YOLO("yolo26s-seg.pt")  # ultralytics downloads it into the working directory
        shutil.move("yolo26s-seg.pt", target)
    print("YOLO26s-seg done")


if __name__ == "__main__":
    main()
