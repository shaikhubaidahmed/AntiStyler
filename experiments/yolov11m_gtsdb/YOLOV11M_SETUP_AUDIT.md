============================================================
YOLOv11-M SETUP AUDIT
============================================================

Model:
YOLOv11-M

Official pretrained checkpoint:
/home/ms/Desktop/AntiStyler/yolo11m.pt

Checkpoint SHA256:
d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95

Checkpoint size:
38.80 MB

Architecture:
PASS

Pretrained initialization:
PASS

Parameter count:
20114688

============================================================
DATASET
============================================================

Dataset:
GTSDB

Classes:
46

Class ID 14:
go left

Class mapping:
PASS

54 test images:
PASS

82 test annotations:
PASS

Same complete test split:
PASS

416 input:
PASS

============================================================
ENVIRONMENT
============================================================

Python:
3.11.8

PyTorch:
2.2.1+cu121

Torchvision:
0.17.1

Ultralytics:
8.3.11

CUDA:
12.1

GPU:
Quadro RTX 4000

VRAM:
7.78 GB

CUDA inference:
PASS

============================================================
COMPATIBILITY
============================================================

YOLOv11-M inference:
PASS

COCO evaluation:
PASS

DPatch:
PASS

AntiStyler:
PASS

Training feasibility:
PASS

============================================================
RESEARCH TARGET
============================================================

Future DPatch target class ID:
14

Dataset-defined class:
go left

Patch:
100 × 100

Location:
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
CRITICAL BLOCKERS
============================================================

NONE

============================================================
FINAL STATUS
============================================================

Dataset compatibility:
PASS

Class mapping:
PASS

Checkpoint:
PASS

GPU:
PASS

DPatch compatibility:
PASS

AntiStyler compatibility:
PASS

COCO compatibility:
PASS

Reproducibility:
PASS

Overall setup:
PASS
