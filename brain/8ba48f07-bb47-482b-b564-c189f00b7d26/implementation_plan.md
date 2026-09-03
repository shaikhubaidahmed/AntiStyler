# Implementation Plan: YOLOv9-M Corrected DPatch Attack

## Goal
Implement and evaluate the spatial DPatch attack on the frozen YOLOv9-M GTSDB baseline, using the exact same constraints and target class (14: 'go right') as previous experiments.

## Proposed Changes

### 1. `attacks/yolov9_patch_attack.py` [NEW]
Create a YOLOv9-M specific DPatch attack class.
- Load the official WongKinYiu `DualDDetect` model in eval mode.
- Access the raw un-decoded feature maps from the forward pass tuple `preds[1][0]` (which corresponds to `d1`, the Main branch spatial outputs of shape `[B, 64+nc, H, W]`).
- Calculate `min_x`, `min_y` based on the same patch location as v7/v8.
- Apply a spatial grid mask corresponding to `strides=[8, 16, 32]`.
- Extract `cls_logits = di[:, 64 + target_class : 64 + target_class + 1, :, :]`.
- Optimize the physical patch to maximize `cls_prob * mask` for 200 epochs using Adam.

### 2. `scripts/yolov9_dpatch_experiment.py` [NEW]
Create a master script to run the experiment.
- Load the clean GTSDB test set (54 images).
- Verify frozen checkpoint SHA256 (`645721eea8b61c415f3b965ffa275d87de1ed0509bd844645656fb6ef124fb5c`).
- Run a small-scale validation on 1 image to verify target class score manipulation at the physical location.
- Generate the full 54 attacked images into `experiments/yolov9m_gtsdb/dpatch/`.
- Evaluate the attacked images via `third_party/yolov9/val_dual.py` or directly by loading the model.
- Generate the `YOLOV9M_DPATCH_REPORT.md`.

## Verification Plan
- Checkpoint SHA256 remains unchanged before and after.
- Attack generates 54 images.
- Attack successfully produces degradation in the primary metric `mAP@0.50` without tuning parameters (maintaining the strict scientific integrity requirement).

> [!IMPORTANT]
> The DPatch parameters (200 epochs, patch size 100, target class 14) will be completely preserved from YOLOv8L.

## Open Questions
- None. The prompt is extremely prescriptive and follows the previous pattern perfectly.
