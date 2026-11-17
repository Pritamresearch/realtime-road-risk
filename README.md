# Real-Time Road Risk Assessment from a Single Camera

Real-time road risk assessment from a single camera: DA3 metric depth + YOLO26-seg + SegFormer-B2, late fusion, live dashboard.

---

## 1. Paper Link, DOI & Citation

- **Publication Status**: Engineering research prototype and evaluation benchmark. *(No peer-reviewed journal or conference paper DOI yet).*
- **Author**: Pritam Chakraborty (pritam23@ieee.org)

```bibtex
@software{chakraborty2026realtime_road_risk,
  author  = {Chakraborty, Pritam},
  title   = {Real-time Road Risk Assessment from a Single Camera: Metric Depth and Instance Segmentation with Late Fusion},
  year    = {2026},
  url     = {https://github.com/Pritamresearch/realtime-road-risk}
}
```

---

## 2. Results

Evaluated on **242 annotated frames across 8 driving clips** (Bolpur, Sikkim, Dhaka day/night, Bhubaneswar night, Malaysia, BDD100K, highway) via leave-one-clip-out (LOCO) cross-validation and two-stage bootstrap resampling (2,000 draws, 95% confidence intervals):

| Method | Top-1 Object Hit [95% CI] | Minus Box Height | Frame AUROC |
| :--- | :---: | :---: | :---: |
| **This system, LOCO-selected (held out)** | **0.430** [0.250, 0.588] | -0.035 [-0.119, +0.023] | **0.964** |
| Box height heuristic (YOLO, no depth) | 0.465 [0.283, 0.630] | — | 0.936 |
| Box bottom heuristic (YOLO, no depth) | 0.430 [0.241, 0.592] | -0.035 [-0.117, +0.014] | 0.846 |
| Class weight only (YOLO, no depth) | 0.394 [0.196, 0.636] | -0.070 [-0.158, +0.071] | 0.868 |
| First version (SegFormer regions, nearest in corridor) | 0.218 [0.068, 0.343] | -0.246 [-0.370, -0.146] | 0.871 |

- **Top-1 Object Hit**: The predicted most critical object has bounding-box IoU $\ge 0.5$ with a human attention box.
- **Frame AUROC**: Discrimination between safety-critical frames and clear frames.
- **Alarm Performance**: NEAR / MID triggers on 86% of critical frames and only 10% of clear frames.

---

## 3. Setup

### Python and Framework Versions
- **Python**: `>= 3.10` (tested on Python 3.12)
- **PyTorch**: `>= 2.0` (tested with CUDA 12.1)
- **Dependencies**: Ultralytics YOLO26, HuggingFace Transformers, OpenCV, Depth Anything 3

### Installation
```bash
git clone https://github.com/Pritamresearch/realtime-road-risk.git
cd realtime-road-risk

# Create virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Install Depth Anything 3 from source
git clone --depth 1 https://github.com/ByteDance-Seed/depth-anything-3 third_party/depth-anything-3
pip install --no-deps -e third_party/depth-anything-3
pip install addict moviepy==1.0.3 trimesh imageio plyfile pycolmap evo tqdm
```

#### `requirements.txt`
```text
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
ultralytics>=8.3.0
transformers>=4.35.0
numpy>=1.24.0
scipy>=1.10.0
pyyaml>=6.0
pillow>=9.5.0
matplotlib>=3.7.0
```

---

## 4. Dataset Instructions

The live pipeline operates on any camera feed or video without pre-downloaded data. For benchmark evaluation and replication:

### Expected Directory Layout
```text
data/
├── videos/                  # Evaluation video clips (e.g., sikim.mp4, bdd100k/00a0f008-3c67908e.mov)
├── annotations/             # Per-clip bounding box CSV annotations (<clip>.csv)
└── bdd_oia/                 # Optional BDD-OIA dataset release
    └── lastframe/
        ├── train_25k_images_actions.json
        └── data/
```

Reference clips and label mapping are configured in [`eval_clips.yaml`](eval_clips.yaml).

---

## 5. Training and Evaluation Commands

All commands below exist directly in the repository.

### Live In-Vehicle Dashboard Execution
```bash
# Research layout: camera, corridor, SegFormer-B2, DA3 metric depth, driver alert card
python run.py path/to/drive.mp4

# In-vehicle driver view: camera + driver alert column only
python run.py path/to/drive.mp4 --view driver

# Live webcam stream
python run.py 0
```

### Single Frame Inference (CLI & API)
```bash
# Predict on single frame and save dashboard image + decision JSON
python predict_frame.py path/to/frame.jpg --out out/frame_risk.png --json out/frame_risk.json
```

### Evaluation Reproduction
```bash
# 1. Cache model inference on annotated frames
python eval_cache.py

# 2. Run 120 pre-declared variants, LOCO selection, bootstrap intervals
python evaluate.py

# 3. Generate diagnostic figures
python make_figures.py
```

---

## 6. Pretrained Weights

Pretrained model checkpoints can be downloaded via the included script:
```bash
python download_weights.py
```
This downloads:
- `weights/da3metric-large/` (Depth Anything 3 Metric-Large)
- `weights/segformer-b2-cityscapes/` (SegFormer-B2 Cityscapes)
- `weights/yolo26s-seg.pt` (YOLO26s instance segmentation)

---

## 7. Hardware Tested

Taken directly from Section 9 ("Speed") of the repository benchmarks:
- **Device**: NVIDIA RTX 3050 6 GB Laptop GPU
- **Execution Mode**: Tested on battery power, capped at 35 W and 1177 MHz (fp16 matmul ~2.9 TFLOPS)
- **Latency Breakdown**:
  - DA3 Metric Depth (392 px input): ~264 ms
  - SegFormer-B2 Segmentation: ~120 ms
  - YOLO26s-seg Detection: ~30 ms
  - Two-stage overlapped fusion + render: ~5 ms (GPU) + ~40 ms (display)
- **Throughput**: ~2.3 - 2.4 FPS on battery power (expected higher when connected to AC power)

---

## 8. Licence

Licence: To be confirmed by author. (The codebase currently references GNU AGPL-3.0 due to the upstream YOLO26 dependency; please contact Pritam Chakraborty regarding licensing terms).
