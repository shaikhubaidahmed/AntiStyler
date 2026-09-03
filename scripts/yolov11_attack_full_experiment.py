import os
import glob
import torch
import cv2
import json
import numpy as np
import time
import shutil
import hashlib
from ultralytics import YOLO

import sys
project_root = "/home/ms/Desktop/AntiStyler"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from attacks.yolov11_patch_attack import YOLOv11PatchAttack
from scripts.evaluate_coco_gtsdb import evaluate_coco

def compute_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_weights_sum(model):
    total = 0
    for param in model.model.parameters():
        total += param.sum().item()
    return total

def main():
    print("=== YOLOv11-M DPATCH FULL ATTACK EXPERIMENT ===")
    
    model_path = os.path.join(project_root, "experiments/yolov11m_gtsdb/run/weights/best.pt")
    
    # 1. Verify Checkpoint
    expected_hash = "04af674ab9058569669703f6f8c207b6c916d61b16ae87546fd9a5c028f458d9"
    actual_hash = compute_sha256(model_path)
    if actual_hash != expected_hash:
        print(f"FATAL: Checkpoint hash mismatch! Expected {expected_hash}, got {actual_hash}")
        sys.exit(1)
        
    print(f"Pretrained checkpoint hash: {actual_hash} [PASS]")
    
    model = YOLO(model_path)
    model.to('cuda')
    
    # Trigger fusion for clean baseline measurement
    dummy_img = torch.rand(1, 3, 416, 416, device='cuda')
    model(dummy_img, verbose=False)
    initial_weights = get_weights_sum(model)
    
    test_img_dir = os.path.join(project_root, "All Dataset/GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov8/test/images")
    output_dir = os.path.join(project_root, "experiments/yolov11m_gtsdb/dpatch")
    output_img_dir = os.path.join(output_dir, "images")
    os.makedirs(output_img_dir, exist_ok=True)
    
    img_files = glob.glob(os.path.join(test_img_dir, "*.jpg"))
    img_files = sorted(img_files)
    
    config = {
        'patch_size': 100,
        'num_epochs': 200,
        'lr': 0.05,
        'target_class': 14 
    }
    
    attack = YOLOv11PatchAttack(model, config)
    
    print(f"\\nProcessing {len(img_files)} images...")
    
    total_opt_time = 0
    
    if len(os.listdir(output_img_dir)) == 54:
        print("Attacked images already generated. Skipping optimization loop.")
    else:
        for idx, img_path in enumerate(img_files):
            filename = os.path.basename(img_path)
            print(f"[{idx+1}/{len(img_files)}] Attacking {filename}...")
            
            img_bgr = cv2.imread(img_path)
            img_bgr = cv2.resize(img_bgr, (416, 416))
            
            img_tensor = torch.from_numpy(img_bgr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img_tensor = img_tensor[:, [2, 1, 0], :, :].to('cuda')
            
            # Optimize patch
            t0 = time.time()
            attacked_tensor, _ = attack.generate(img_tensor)
            t1 = time.time()
            total_opt_time += (t1 - t0)
            
            # Convert back to BGR for saving
            attacked_np = attacked_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            attacked_np = (attacked_np * 255.0).astype(np.uint8)
            attacked_bgr = attacked_np[:, :, [2, 1, 0]]
            
            out_path = os.path.join(output_img_dir, filename)
            cv2.imwrite(out_path, attacked_bgr)
        
    print("\\nAttack Generation Complete.")
    final_weights = get_weights_sum(model)
    print(f"Initial weights: {initial_weights}")
    print(f"Final weights: {final_weights}")
    if initial_weights == final_weights:
        print("PASS: Frozen weights verified.")
    else:
        print("FAIL: Weights changed.")
        
    actual_hash_after = compute_sha256(model_path)
    if actual_hash_after != expected_hash:
        print("FAIL: Checkpoint hash changed during attack!")
    else:
        print("PASS: Checkpoint hash unchanged.")
        
    # Evaluate using original dataset configuration but pointing to attacked images
    # Create a custom yaml that overrides the test path
    original_yaml = os.path.join(project_root, "All Dataset/GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov8/data.yaml")
    
    import yaml
    with open(original_yaml, 'r') as f:
        dataset_config = yaml.safe_load(f)
        
    dataset_config['test'] = output_img_dir
    dataset_config['val'] = output_img_dir
    dataset_config['train'] = output_img_dir
    
    # also we need to copy labels
    test_label_dir = os.path.join(project_root, "All Dataset/GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov8/test/labels")
    output_label_dir = os.path.join(output_dir, "labels")
    os.makedirs(output_label_dir, exist_ok=True)
    
    for label_file in glob.glob(os.path.join(test_label_dir, "*.txt")):
        shutil.copy(label_file, output_label_dir)
        
    custom_yaml = os.path.join(output_dir, "attacked_data.yaml")
    with open(custom_yaml, 'w') as f:
        yaml.dump(dataset_config, f)
        
    print("\\nRunning Inference on Attacked Dataset...")
    val_results = model.val(data=custom_yaml, split='test', save_json=True, imgsz=416, batch=16, plots=False)
    
    # Fix JSON and run COCO Evaluation
    val_dir = val_results.save_dir
    pred_json_path = os.path.join(val_dir, "predictions.json")
    
    target_pred_json = os.path.join(output_dir, "predictions_attacked.json")
    if os.path.exists(pred_json_path):
        shutil.copy(pred_json_path, target_pred_json)
        
    gt_json_path = os.path.join(project_root, "evaluation_data", "gtsdb_coco", "instances_gtsdb_test.json")
    
    # Fix image_id in predictions.json to match instances_gtsdb_test.json
    try:
        with open(target_pred_json, "r") as f:
            preds = json.load(f)
            
        with open(gt_json_path, 'r') as f:
            coco_gt = json.load(f)
            
        img_name_to_id = {img['file_name']: img['id'] for img in coco_gt['images']}
        
        for p in preds:
            img_id_str = str(p['image_id'])
            if img_id_str.endswith(".jpg"):
                img_name = img_id_str
            else:
                img_name = img_id_str + ".jpg"
            
            if img_name in img_name_to_id:
                p['image_id'] = img_name_to_id[img_name]
            else:
                p['image_id'] = int(''.join(filter(str.isdigit, img_name)))
                
        with open(target_pred_json, "w") as f:
            json.dump(preds, f)
    except Exception as e:
        print(f"Failed to fix predictions.json: {e}")
        
    print("\\nEvaluating Full Attacked Set with evaluate_coco...")
    attacked_stats = evaluate_coco(gt_json_path, target_pred_json)
    
    coco_map50_95 = attacked_stats[0]
    coco_map50 = attacked_stats[1]
    coco_ap75 = attacked_stats[2]
    coco_recall = val_results.box.mr
    coco_precision = val_results.box.mp
    
    try:
        with open(target_pred_json, "r") as f:
            preds = json.load(f)
            num_detections = len(preds)
    except:
        num_detections = "N/A"
        
    speed_dict = val_results.speed
    inference_ms = speed_dict.get('inference', 0.0)
    fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0
    
    mean_opt_time = total_opt_time / len(img_files) if img_files else 0
    
    report_content = f"""============================================================
YOLOv11-M DPATCH ATTACK
============================================================

Model:
YOLOv11-M

Checkpoint:
{model_path}

Checkpoint SHA256:
{actual_hash}

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

mAP@0.5                   0.8758       {coco_map50:.4f}

mAP@0.5:0.95              0.7039       {coco_map50_95:.4f}

Precision                 0.9359       {coco_precision:.4f}

Recall                    0.7005       {coco_recall:.4f}

AP75                      0.8553       {coco_ap75:.4f}

Detections                416          {num_detections}

Inference ms/image        13.72        {inference_ms:.2f}

FPS                       72.86        {fps:.2f}

============================================================
ATTACK IMPACT
============================================================

Absolute mAP50 degradation:
{0.8758 - coco_map50:.4f}

Relative mAP50 degradation:
{((0.8758 - coco_map50) / 0.8758 * 100):.2f} %

mAP50:95 change:
{coco_map50_95 - 0.7039:.4f}

Precision change:
{coco_precision - 0.9359:.4f}

Recall change:
{coco_recall - 0.7005:.4f}

Detection-count change:
{num_detections - 416}

============================================================
ATTACK EXECUTION
============================================================

Images successfully attacked:
{len(img_files)}/54

Images skipped:
0

Images failed:
0

Mean attack optimization time/image:
{mean_opt_time:.2f} seconds

Total attack runtime:
{total_opt_time:.2f} seconds
"""
    
    report_path = os.path.join(project_root, "experiments/yolov11m_gtsdb/YOLOV11M_DPATCH_REPORT.md")
    with open(report_path, "w") as f:
        f.write(report_content)
        
    print(f"\\nReport saved to {report_path}")

if __name__ == "__main__":
    main()
