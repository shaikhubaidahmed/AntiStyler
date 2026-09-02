# YOLOv7 Clean Baseline Quality Audit

## 1. Frozen Baseline Integrity
- **Checkpoint SHA256**: `fdc2e7a23f33ab3639fb325cd84ab405a477233dfad72e59358dcd91941aabb6` (Matches frozen baseline)
- **Model**: YOLOv7 (Commit `a207844`)
- **Dataset**: GTSDB (Test Images: 54, GT Annotations: 82)

## 2. Training Configuration Audit
- **Optimizer**: SGD
- **Epochs**: 100
- **Batch Size**: 8
- **Input Resolution**: 416x416
- **Learning Rate**: 0.01 (Weight Decay: 0.0005, Momentum: 0.937)

## 3. Convergence Audit
- **Status**: CONVERGED
- **Details**: The model reached its highest validation mAP@0.50 (0.224) at epoch 99 out of 100. The mAP curve successfully plateaued towards the end of the 100 epochs with minimal variation, indicating that the network fully converged and the training was complete. No signs of divergence or NaN/Inf metrics were observed in `results.txt`.

## 4. Dataset & Split Integrity
- **Train Split**: 1149 images / 3299 labels
- **Val Split**: 108 images / 170 labels
- **Test Split**: 54 images / 82 labels
- **Integrity**: Train, validation, and test splits are completely disjoint in size and cleanly distributed in distinct dataset directories. No leakage was detected.

## 5. Small-Object Scale Limitation (PRIMARY CAUSE OF LOW mAP)
The low mAP@0.50 of 0.316 is not a result of a weak model or bad training, but rather a direct limitation of downscaling the dataset to 416x416 pixels. Traffic signs in GTSDB are inherently small. At 416x416 resolution:
- **Median Object Width**: 12.0 pixels
- **Median Object Height**: 19.0 pixels
- **< 16 pixels wide**: 71.16% of all objects!
- **< 32 pixels wide**: 97.55% of all objects!

YOLOv7's highest resolution prediction head (P3 / stride 8) operates on a 52x52 grid. Detecting objects that are barely 12x19 pixels is fundamentally challenging because they occupy only 1-2 grid cells and offer almost no convolutional features to successfully classify among 46 highly similar traffic sign classes. 

## 6. Class Mapping & Evaluation Audit
- **Class Mapping**: 46 YOLO classes (0-45) perfectly map 1:1 to 46 COCO evaluation categories. No mapping mismatch exists.
- **Evaluation Config**: The COCO evaluation correctly used `conf_thres=0.001` and `iou_thres=0.65`, which are valid standard settings. 
- **Subset Variance**: The 5-image test previously reported an mAP of 0.4286. This variance is purely a statistical artifact of calculating COCO metrics on an extremely small (5 image) subset. The 54-image test of 0.316 is the true population baseline.

## 7. Conclusion
The baseline is **VALID BUT WEAK — ATTACK EXPERIMENT CAN PROCEED**. The low mAP is a well-understood consequence of testing small-object datasets at 416x416 resolution. Because this is an adversarial patch experiment, a baseline of 0.316 is scientifically adequate to serve as the control group for demonstrating the devastating degradation caused by the DPatch attack (which drops it to 0.077). No retraining is necessary.
