# YOLOv7 Checkpoint Setup

## Checkpoint Properties
- **Model**: YOLOv7 (standard)
- **Filename**: `yolov7.pt`
- **Source**: https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
- **Architecture**: YOLOv7 with IDetect head, 3 output scales (P3/P4/P5)
- **Pretrained on**: COCO (80 classes)
- **File Size**: 75,587,165 bytes (72 MB)
- **SHA-256**: `d1ac3c74eb96a3eec77949c0f37a06bc272756606ff011d9353f7abff4e0c71d`

## Local Location
`/home/ms/Desktop/AntiStyler/third_party/yolov7/yolov7.pt`

## Validation
- Loaded successfully via `attempt_load()` from official YOLOv7 source
- Stride: 32
- Classes: 80 (COCO)
- Inference verified on GTSDB test images (detects COCO objects as expected)
- Inference speed: ~14.24 ms per image (70.2 FPS) on Quadro RTX 4000

## Important Notes
- This is a COCO-pretrained checkpoint. It must be **fine-tuned** on GTSDB to detect traffic signs.
- The COCO model has 80 classes; GTSDB has 46 classes. The detection head will be replaced during fine-tuning.
