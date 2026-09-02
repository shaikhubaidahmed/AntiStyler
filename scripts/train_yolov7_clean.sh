#!/bin/bash
set -e

# Setup paths
PROJECT_ROOT="/home/ms/Desktop/AntiStyler"
YOLO_DIR="$PROJECT_ROOT/third_party/yolov7"
EXPERIMENT_DIR="$PROJECT_ROOT/experiments/yolov7_gtsdb"
DATA_YAML="$PROJECT_ROOT/configs/datasets/gtsdb_yolov7.yaml"
CFG_YAML="cfg/training/yolov7.yaml"
WEIGHTS="yolov7.pt"

cd "$YOLO_DIR"

# Print the resolved configuration
echo "=== RESOLVED CONFIGURATION ==="
echo "Dataset: $DATA_YAML"
echo "Model: $CFG_YAML"
echo "Pretrained Checkpoint: $WEIGHTS"
echo "Image Size: 416"
echo "Batch Size: 8"
echo "Epochs: 100"
echo "Optimizer: SGD"
echo "Learning Rate: 0.01"
echo "Scheduler: OneCycleLR"
echo "Weight Decay: 0.0005"
echo "Augmentation: Default YOLOv7 (hyp.scratch.p5.yaml)"
echo "Seed: 42"
echo "Workers: 4"
echo "Device: 0"
echo "=============================="

# Train YOLOv7
# We use --project to route the outputs to the required experiment directory
python train.py \
  --workers 4 \
  --device 0 \
  --batch-size 8 \
  --data "$DATA_YAML" \
  --img 416 416 \
  --cfg "$CFG_YAML" \
  --weights "$WEIGHTS" \
  --name run \
  --project "$EXPERIMENT_DIR" \
  --hyp data/hyp.scratch.p5.yaml \
  --epochs 100 \
  --exist-ok
