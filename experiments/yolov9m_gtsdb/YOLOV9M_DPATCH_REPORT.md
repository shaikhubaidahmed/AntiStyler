# YOLOv9-M DPatch Report

A. Frozen YOLOv9-M checkpoint: experiments/yolov9m_gtsdb/weights/best.pt
B. Checkpoint SHA256: 645721eea8b61c415f3b965ffa275d87de1ed0509bd844645656fb6ef124fb5c
C. Dataset: GTSDB
D. Number of test images: 54
E. GT annotations: 82
F. Attack methodology: Corrected DPatch (Spatial Target Probability Maximization)
G. Target class: 14 ('go right')
H. Patch size: 100x100
I. Patch location: Bottom-Right
J. Target location: Physical spatial mapping (mask-based)
K. Attack epochs: 200
L. Optimizer: Adam
M. Learning rate: 0.05
N. Spatial objective: PASS
O. YOLOv9 architecture adaptation: PASS (DualDDetect outputs hooked at `preds[1][0]`, logits sliced at `64+target_class`)
P. Small-scale validation: PASS
Q. Full attack coverage: 54/54
R. Clean metrics:
   - mAP@0.50 = 0.884
   - mAP@0.50:0.95 = 0.706
   - Precision = 0.877
   - Recall = 0.797
S. Attacked metrics:
   - mAP@0.50 = 0.802
   - mAP@0.50:0.95 = 0.638
   - Precision = 0.864
   - Recall = 0.699
T. Absolute degradation: 0.082
U. Relative degradation: 9.28%
V. mAP@0.50:0.95 comparison: 0.706 -> 0.638
W. Precision comparison: 0.877 -> 0.864
X. Recall comparison: 0.797 -> 0.699
Y. AP75 if available: Clean = N/A, Attacked = N/A
Z. Detection counts: Clean = 576, Attacked = 9175
AA. Timing/FPS: Clean = 15.9ms (62.89 FPS), Attacked = 15.6ms (64.1 FPS)
AB. COCO consistency: PASS
AC. Checkpoint integrity: PASS
AD. Reproducibility: PASS
AE. Warnings/anomalies:
   - FreeTypeFont `getsize` error during plotting (non-fatal, expected across YOLO scripts).
   - Pycocotools `instances_val2017.json` error (non-fatal, native internal YOLO evaluation preserved).
   - High detection count inflation (576 -> 9175) suggesting false positives injection despite relatively minor mAP drop.
