#!/bin/bash
set -e
cd third_party/yolov9
python val_dual.py \
  --data "/home/ms/Desktop/AntiStyler/All Dataset/GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov9/data.yaml" \
  --weights ../../experiments/yolov9m_gtsdb/weights/best.pt \
  --task test \
  --img 416 \
  --save-json \
  --name yolov9m_test_eval \
  --project ../../experiments/yolov9m_gtsdb \
  --exist-ok
