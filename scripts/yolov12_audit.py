import os
import sys
import torch
import torchvision
import platform
import hashlib
import json
import yaml
import time
import subprocess

PROJECT_ROOT = "/home/ms/Desktop/AntiStyler"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "yolov12"))

from ultralytics import YOLO

def get_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def main():
    print("=== YOLOv12-M SETUP AUDIT ===")
    
    # 1. Environment
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Torchvision: {torchvision.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")
    if cuda_avail:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"VRAM: {total_memory:.2f} GB")
        
    try:
        from ultralytics import __version__ as ul_version
        print(f"Implementation (Ultralytics Fork): {ul_version}")
    except:
        pass
        
    # 2. Checkpoint
    ckpt_path = os.path.join(PROJECT_ROOT, "yolo12m.pt")
    if not os.path.exists(ckpt_path):
        print("ERROR: Checkpoint not found")
        sys.exit(1)
        
    file_size = os.path.getsize(ckpt_path) / (1024 * 1024)
    ckpt_hash = get_hash(ckpt_path)
    
    print(f"Checkpoint size: {file_size:.2f} MB")
    print(f"Checkpoint SHA256: {ckpt_hash}")
    
    # 3. Model Loading
    print("Loading YOLOv12-M...")
    try:
        model = YOLO(ckpt_path)
        model.to('cuda')
        
        pytorch_total_params = sum(p.numel() for p in model.model.parameters())
        print(f"Parameter count: {pytorch_total_params}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)
        
    # 4. Smoke Test
    test_img = os.path.join(PROJECT_ROOT, "All Dataset/GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov8/test/images/00026_jpg.rf.4c55e8a72c668d5a0c4a948580757cd9.jpg")
    print(f"Running inference on {test_img}...")
    try:
        start_time = time.time()
        results = model(test_img, imgsz=416, verbose=False)
        inf_time = time.time() - start_time
        print(f"Inference successful: {inf_time:.3f} seconds")
        print(f"Boxes detected: {len(results[0].boxes)}")
    except Exception as e:
        print(f"ERROR in inference: {e}")
        
    # 5. Dataset Validation
    dataset_yaml = os.path.join(PROJECT_ROOT, "All Dataset/GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov8/data.yaml")
    with open(dataset_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)
        
    print(f"Dataset classes: {data_cfg['nc']}")
    print(f"Class 14 Name: {data_cfg['names'][14]}")
    
    gt_json = os.path.join(PROJECT_ROOT, "evaluation_data/gtsdb_coco/instances_gtsdb_test.json")
    with open(gt_json, 'r') as f:
        gt_data = json.load(f)
    print(f"Test images in GT: {len(gt_data['images'])}")
    print(f"GT annotations: {len(gt_data['annotations'])}")
    
if __name__ == "__main__":
    main()
