============================================================
YOLOv11-M CLEAN BASELINE
============================================================

Model:
YOLOv11-M

Pretrained checkpoint:
/home/ms/Desktop/AntiStyler/yolo11m.pt

Pretrained SHA256:
d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95

Best checkpoint:
/home/ms/Desktop/AntiStyler/experiments/yolov11m_gtsdb/run/weights/best.pt

Best checkpoint SHA256:
04af674ab9058569669703f6f8c207b6c916d61b16ae87546fd9a5c028f458d9

Dataset:
GTSDB

Classes:
46

Test images:
54

GT annotations:
82

Input:
416 × 416

Epochs:
100

Actual batch size:
16

============================================================
MAIN METRICS
============================================================

Metric                    YOLOv11-M Clean

mAP@0.5                   0.8758

mAP@0.5:0.95              0.7039

Precision                 0.9359

Recall                    0.7005

AP75                      0.855325774512935

Detections                416

Inference ms/image        13.72

FPS                       72.86

============================================================
COCO VALIDATION
============================================================

Native mAP50:
0.8751

COCO mAP50:
0.8758

Difference:
0.0007

Consistency:
PASS

============================================================
PER-CLASS
============================================================

Per-class metrics:
runs/detect/val7/results.csv

Class 14:
go left

Class 14 AP50:
N/A (See results.csv for detailed per-class)

============================================================
TRAINING
============================================================

Training duration:
0.49 hours

Best epoch:
N/A (using early stopping / best weights)

Final epoch:
100

GPU:
NVIDIA Quadro RTX 4000

Peak VRAM:
N/A
