#!/bin/bash
set -e

cd third_party/yolov9

# Ensure the dataset yaml has the correct classes and paths (done)

echo "Starting YOLOv9-M Training..."

python train_dual.py \
  --workers 4 \
  --device 0 \
  --batch-size 16 \
  --data "/home/ms/Desktop/AntiStyler/All Dataset/GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov9/data.yaml" \
  --img 416 \
  --cfg models/detect/yolov9-m.yaml \
  --weights ../../yolov9-m.pt \
  --name yolov9m_gtsdb \
  --hyp data/hyps/hyp.scratch-high.yaml \
  --epochs 100 \
  --project ../../experiments \
  --exist-ok

echo "Training completed!"
