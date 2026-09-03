============================================================
YOLOv11-M DPATCH ATTACK
============================================================

Model:
YOLOv11-M

Checkpoint:
/home/ms/Desktop/AntiStyler/experiments/yolov11m_gtsdb/run/weights/best.pt

Checkpoint SHA256:
04af674ab9058569669703f6f8c207b6c916d61b16ae87546fd9a5c028f458d9

Dataset:
GTSDB

Test images:
54

GT annotations:
82

Target class ID:
14

Target class:
go left

Patch:
100 × 100

Patch location:
bottom-right

Optimization:
200 epochs

Optimizer:
Adam

Learning rate:
0.05

Spatial objective:
YES

============================================================
MAIN RESULTS
============================================================

Metric                    Clean       Attacked

mAP@0.5                   0.8758       0.8066

mAP@0.5:0.95              0.7039       0.6267

Precision                 0.9359       0.9269

Recall                    0.7005       0.6407

AP75                      0.8553       0.7459

Detections                416          2084

Inference ms/image        13.72        10.39

FPS                       72.86        96.26

============================================================
ATTACK IMPACT
============================================================

Absolute mAP50 degradation:
0.0692

Relative mAP50 degradation:
7.91 %

mAP50:95 change:
-0.0772

Precision change:
-0.0090

Recall change:
-0.0598

Detection-count change:
1668

============================================================
ATTACK EXECUTION
============================================================

Images successfully attacked:
54/54

Images skipped:
0

Images failed:
0

Mean attack optimization time/image:
0.00 seconds

Total attack runtime:
0.00 seconds
