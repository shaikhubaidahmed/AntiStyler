# YOLOv7 DPatch Correction Documentation

## Source Basis for Corrections
- **Spatial Masking**: The `AntiStyler_Demo.ipynb` notebook implements DPatch by enforcing a targeted supervised loss on a specific target location (`'boxes': attack_config['target_location']`). This specifically directs the optimizer to alter the predictions at that physical coordinate rather than aggregating loss indiscriminately across the image.
- **200 Epochs**: The `AntiStyler_Demo.ipynb` notebook explicitly defines `"num_epochs" : 200` in `attack_config`.

## Architectural Adaptation
YOLOv7 operates via grid cells across three different spatial scales, unlike Faster R-CNN's unified Region Proposal Network mechanism. We mapped the target physical patch bounding box to YOLOv7's internal prediction grids.

## Spatial-Mask Derivation & Multi-Scale Handling
YOLOv7 has detection heads at strides of 8, 16, and 32 relative to the 416x416 input image. 
- Grid sizes: 52x52, 26x26, 13x13.
- The physical patch coordinates `[min_y, min_x, min_y + patch_size, min_x + patch_size]` are divided by each stride $s \in \{8, 16, 32\}$ and passed through `floor()` and `ceil()` to compute the exact grid indices affected by the patch.
- A spatial binary mask is constructed for each stride, matching the dimensions of YOLOv7's prediction grids.

## Affected Grid Cells
Only the anchor grid cells bounded by `[start_y:end_y, start_x:end_x]` derived for each scale are included in the attack objective summation.

## Attack Objective
The loss function maximizes `objectness * target_class_probability` exclusively for the grid cells defined by the spatial mask. The loss is normalized by the active mask sum to maintain proper gradient scaling. 

## Attack Parameters
- **Optimizer**: Adam
- **Learning rate**: 0.1
- **Epochs**: 200
- **Patch size**: 100x100
- **Patch location**: Bottom-right quadrant (`min_y = (H+100)/2`, `min_x = (W+100)/2`)
- **Random seed**: Not explicitly fixed in `yolov7_patch_attack.py` (relies on PyTorch default state)
- **Image resolution**: 416x416
- **Evaluation**: COCO API mAP@0.50 (conf_thres=0.001, nms_iou=0.65)
- **Checkpoint SHA256**: `fdc2e7a23f33ab3639fb325cd84ab405a477233dfad72e59358dcd91941aabb6`
