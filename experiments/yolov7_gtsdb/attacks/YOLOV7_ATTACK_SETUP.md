# YOLOv7 Adversarial Patch Attack Setup

## Environment & Hardware
- **Python version**: 3.11.8
- **PyTorch version**: 2.2.1+cu121
- **CUDA version**: 12.1
- **GPU**: NVIDIA Quadro RTX 4000

## Target Model
- **YOLOv7 Commit**: `a207844`
- **Checkpoint Path**: `experiments/yolov7_gtsdb/run/weights/best.pt`
- **Checkpoint SHA256**: `fdc2e7a23f33ab3639fb325cd84ab405a477233dfad72e59358dcd91941aabb6`

## Target Dataset
- **Dataset Split**: GTSDB `test` split
- **Random seed**: Not strictly applicable for standard sequential inference, but the PyTorch environment uses the default PyTorch deterministic seed state during evaluation.

## Attack Implementation
- **Implementation**: Custom PyTorch module `YOLOv7PatchAttack` (based on DPatch/Adam methodology from AntiStyler notebook)
- **Code Path**: `attacks/yolov7_patch_attack.py`
- **Methodology**: The attack optimizes a trainable patch tensor via Adam to maximize the objectness score of a target dummy class (Class 0: ANIMALS). The DPatch spatial objective is adapted to YOLOv7 by computing the exact prediction grid cells that overlap with the physical patch across all three YOLOv7 detection scales (stride 8, 16, 32) and restricting the loss function strictly to those affected grid cells. This forces the model to generate a confident target class prediction at the precise patch location, reproducing the targeted localized loss function of DPatch while accounting for YOLOv7's multi-scale anchor structure.

## Attack Parameters
- **Patch dimensions (`patch_size`)**: 100x100
- **Patch placement**: Fixed to the bottom-right-ish quadrant (`min_y = (H + patch_size) // 2`, `min_x = (W + patch_size) // 2`) clamped to image boundaries, reproducing the logic from `AntiStyler_Demo.ipynb`.
- **Number of attack iterations (`num_epochs`)**: 200 (Matches `AntiStyler_Demo.ipynb`)
- **Attack optimizer**: Adam
- **Learning rate (`lr`)**: 0.1
- **Target class**: 0 (ANIMALS) - acts as a dummy class to force bounding box conflict.

## Inference Parameters
- **Image resolution**: 416x416
- **Confidence threshold**: 0.001
- **NMS IoU**: 0.65
- **Max detections**: 300 (YOLOv7 standard)
- **Preprocessing**: `utils.datasets.letterbox` (stride 32), normalized to [0, 1]. No augmentations during inference/attack optimization.

## Data Integrity
- The original GTSDB images are NOT modified. The attack modifies cloned tensors during runtime.
- Annotations remain untouched.
