# YOLOv8L DPatch Attack Experiment Report

- Dataset: GTSDB (54 images)
- Architecture: YOLOv8L (Frozen Baseline)
- Attack Target Class: 14
- Optimization: 200 epochs, lr=0.05, patch_size=100

## Results
- Clean mAP@0.50: 0.912
- Attacked mAP@0.50: 0.8656
- Attacked mAP@0.50:0.95: 0.6973
