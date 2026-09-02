# YOLOv8L Clean Baseline Report

## A. Model Identity
- **Model**: YOLOv8L
- **Framework**: Ultralytics 8.1.34

## B. Pretrained Checkpoint Identity
- **Pretrained Checkpoint**: `yolov8l.pt`

## C. Pretrained Checkpoint SHA256
- **Hash**: `18218ea4798da042d9862e6029ca9531adbd40ace19b6c9a75e2e28f1adf30cc`

## D. Trained Best Checkpoint Path
- **Path**: `experiments/yolov8l_gtsdb/run/weights/best.pt`

## E. Trained Checkpoint SHA256
- **Hash**: `c9554d26f09f377a048c32ff7ade71baad8e5b90ad678fc43f32a1df74fbaddc`

## F. Dataset Identity
- **Dataset**: GTSDB

## G. Dataset Split
- **Split**: 1149 train, 108 val, 54 test images (Original isolated project split)

## H. Number of Classes
- **Classes**: 46

## I. Class Mapping Verification
- **Status**: PASS (Matches exactly with COCO GT mappings)

## J. Training Configuration
- **Epochs Target**: 100
- **Image Size**: 416
- **Seed**: 42

## K. Actual Batch Size
- **Batch Size**: 16 (Executed successfully without OOM fallback)

## L. Epochs Completed
- **Epochs**: 100

## M. Image Size
- **Resolution**: 416x416

## N. GPU Information
- **GPU**: Quadro RTX 4000

## O. Python Version
- **Python**: 3.11.8

## P. PyTorch Version
- **PyTorch**: 2.2.1+cu121

## Q. CUDA Version
- **CUDA**: 12.1

## R. Ultralytics Version
- **Ultralytics**: 8.1.34

## S. Clean COCO mAP@0.50
- **COCO mAP@0.50**: 0.912

## T. Clean mAP@0.50:0.95
- **COCO mAP@0.50:0.95**: 0.735

## U. Precision
- **Precision**: 0.734 (Ultralytics Internal)

## V. Recall
- **Recall**: 0.857 (Ultralytics Internal)

## W. AP75
- **AP75**: 0.892 (PyCOCOTools)

## X. Per-class metrics
- **Status**: AVAILABLE (Ultralytics Internal metrics printed in logs)

## Y. Confusion Matrix
- **Status**: AVAILABLE (Generated in `experiments/yolov8l_gtsdb/run`)

## Z. Detection Count
- **Detections**: 307 (On 54 test images)

## AA. Inference Time
- **Inference**: 15.2 ms / image

## AB. FPS
- **FPS**: 65.79 FPS

## AC. COCO/Internal Metric Consistency
- **Ultralytics Internal mAP@0.50**: 0.908
- **PyCOCOTools mAP@0.50**: 0.912
- **Absolute Difference**: 0.004
- **Authoritative Metric**: PyCOCOTools (0.912)

## AD. Warnings or Anomalies
- **None**. The YOLOv8L model displayed a massive performance improvement (0.912 mAP) compared to the YOLOv7 baseline (0.316 mAP) on the same 416x416 resolution, indicating superior handling of the GTSDB small-object distributions.
