# Multi-Object Tracking Under Occlusion

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00BFFF)

Occlusion-aware multi-object tracking using **YOLOv8 + custom DeepSORT** with Re-ID, achieving **0 ID Switches** and **96.88% IDF1**.

## Results

| Metric | Baseline | **OcclusionDeepSORT** |
|:-------|:---------|:----------------------|
| MOTA | 93.40% | **93.94%** |
| IDF1 | 91.16% | **96.88%** ✅ |
| ID Switches | 1 | **0** ✅ |

## Setup

```bash
git clone https://github.com/dhruvilmodi111/Multi-object-tracker.git
cd Multi-object-tracker
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Run tracking
python scripts/run_tracking.py \
    --seq_dir data/MOT17/train/MOT17-02-FRCNN \
    --config configs/tuned_config.yaml \
    --output results/output.txt \
    --save_video results/tracked.mp4

# Evaluate (MOTA / IDF1 / IDSW)
python scripts/evaluate.py \
    --results results/output.txt \
    --gt_path data/MOT17/train/MOT17-02-FRCNN/gt/gt.txt
```

## Stack

- **Detection:** YOLOv8n
- **Re-ID:** MobileNetV3 (512-dim embeddings)
- **Tracking:** Custom DeepSORT with multi-stage matching, lost-track buffering (35 frames), and occlusion freeze
- **Evaluation:** [py-motmetrics](https://github.com/cheind/py-motmetrics)
