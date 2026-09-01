# YOLOv7 Baseline Setup

## Dataset

- **Name**: GTSDB (German Traffic Sign Detection Benchmark)
- **Version**: v3-augmented-Roboflow-ACCURATE-model
- **Format**: YOLOv7 PyTorch (YOLO txt)
- **Path**: `All Dataset/GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov7pytorch/`
- **Audit**: See `YOLOV7_DATASET_AUDIT.md`

## Dataset Statistics

| Property | Value |
|----------|-------|
| Train images | 1,149 |
| Validation images | 108 |
| Test images | 54 |
| Total annotations | 3,551 |
| Classes | 46 |
| Image size | 416×416 |
| Annotation format | YOLO txt |
| Annotation errors | 0 |

## YOLOv7 Source
- **Repository**: WongKinYiu/yolov7 (official)
- **Local Path**: `third_party/yolov7/`
- **Commit**: HEAD of main branch (cloned 2026-09-01)
- **Status**: ✅ Available

## YOLOv7 Checkpoint
- **File**: `third_party/yolov7/yolov7.pt`
- **SHA-256**: `d1ac3c74eb96a3eec77949c0f37a06bc272756606ff011d9353f7abff4e0c71d`
- **Pretrained on**: COCO (80 classes)
- **Status**: ✅ Downloaded and verified

## Dependencies
- See `YOLOV7_DEPENDENCY_ANALYSIS.md`
- **Status**: ✅ Compatible (numpy version nominally out of range but functionally verified)

## Training Configuration
- **Config file**: `configs/models/yolov7_gtsdb.yaml`
- **Image size**: 416 (matches dataset)
- **Batch size**: 8 (conservative for 8GB VRAM)
- **Epochs**: 100
- **Optimizer**: SGD (YOLOv7 default)
- **Learning rate**: 0.01 (YOLOv7 default)
- **Seed**: 42
- **Augmentation**: YOLOv7 defaults (dataset already augmented by Roboflow)

## Evaluation Configuration
- **Metric**: mAP@0.5 (standard COCO-style)
- **Confidence threshold**: 0.001 (for evaluation)
- **NMS IoU threshold**: 0.65

## Hardware
- **GPU**: NVIDIA Quadro RTX 4000
- **VRAM**: ~8 GB
- **CUDA**: 12.1
- **PyTorch**: 2.2.1+cu121
- **Inference speed**: 14.24 ms/image (70.2 FPS) at 416×416

## Reproducibility
- Explicit seed: 42
- Fixed image size: 416
- Fixed batch size: 8
- Documented checkpoint hashes
- All configurations under version control

## Smoke Test
- **Status**: ✅ PASS
- **Model loading**: ✅
- **Image loading**: ✅
- **Inference**: ✅
- **Prediction format**: ✅ Valid bboxes, confidences, class IDs
- **NaN/Inf check**: ✅ None detected
- **Visual outputs**: Saved to `debug_outputs/yolov7_smoke_test/`
- **Note**: Detections are COCO classes (car, person, truck) — this is expected since the checkpoint is COCO-pretrained and has not been fine-tuned on GTSDB yet.

## Known Issues
1. **COCO-pretrained checkpoint detects COCO classes, not traffic signs**. Fine-tuning on GTSDB is required before the model can be used for traffic sign detection research.
2. **Duplicate class semantics** in the Roboflow export (e.g., `no overtaking (trucks)` vs `no overtaking -trucks-`). This is a property of the dataset and will not be modified.
3. **numpy version** is 1.26.4 vs YOLOv7's specified `<1.24.0`. Functionally compatible; verified by smoke test.

## Training Readiness

| Checklist Item | Status |
|----------------|--------|
| Dataset valid | ✅ |
| YOLOv7 source available | ✅ |
| Compatible dependencies | ✅ |
| Pretrained checkpoint available | ✅ |
| Model loads | ✅ |
| Dataset loads | ✅ |
| Inference works | ✅ |
| GPU works | ✅ |
| Configuration reproducible | ✅ |

**TRAINING READINESS: READY**
