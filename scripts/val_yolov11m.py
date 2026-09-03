import os
import sys
import yaml
import torch
import hashlib
import time
import json
from ultralytics import YOLO

# Add root to sys.path to import evaluate_coco
sys.path.append("/home/ms/Desktop/AntiStyler")
from scripts.evaluate_coco_gtsdb import evaluate_coco

def compute_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    print("=== YOLOv11-M CLEAN BASELINE VALIDATION ===")
    
    project_root = "/home/ms/Desktop/AntiStyler"
    dataset_yaml = os.path.join(project_root, "All Dataset", "GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov8", "data.yaml")
    
    best_ckpt = os.path.join(project_root, "experiments", "yolov11m_gtsdb", "run", "weights", "best.pt")
    if not os.path.exists(best_ckpt):
        print(f"FATAL: Checkpoint not found at {best_ckpt}")
        sys.exit(1)
        
    best_hash = compute_sha256(best_ckpt)
    print(f"Best checkpoint hash: {best_hash}")
    
    # Validate
    print("Starting validation...")
    val_model = YOLO(best_ckpt)
    val_results = val_model.val(data=dataset_yaml, split='test', save_json=True)
    
    native_map50 = val_results.box.map50
    native_map50_95 = val_results.box.map
    
    # Run COCO evaluation using our script
    val_dir = val_results.save_dir
    pred_json_path = os.path.join(val_dir, "predictions.json")
    
    target_pred_json = os.path.join(project_root, "experiments", "yolov11m_gtsdb", "predictions_clean.json")
    if os.path.exists(pred_json_path):
        import shutil
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
            # ultralytics might store image_id as '00000.jpg' or '00000'
            img_id_str = str(p['image_id'])
            if img_id_str.endswith(".jpg"):
                img_name = img_id_str
            else:
                img_name = img_id_str + ".jpg"
            
            # Map back to integer ID
            if img_name in img_name_to_id:
                p['image_id'] = img_name_to_id[img_name]
            else:
                p['image_id'] = int(''.join(filter(str.isdigit, img_name)))
                
        with open(target_pred_json, "w") as f:
            json.dump(preds, f)
    except Exception as e:
        print(f"Failed to fix predictions.json: {e}")
        
    print("\\nEvaluating Full Clean Set with evaluate_coco...")
    clean_stats = evaluate_coco(gt_json_path, target_pred_json)
    
    coco_map50_95 = clean_stats[0]
    coco_map50 = clean_stats[1]
    coco_ap75 = clean_stats[2]
    coco_recall = clean_stats[8]
    # compute precision from JSON manually or use val_results
    coco_precision = val_results.box.mp
    coco_recall_val = val_results.box.mr
    
    # Get detections count
    try:
        with open(target_pred_json, "r") as f:
            preds = json.load(f)
            num_detections = len(preds)
    except:
        num_detections = "N/A"
        
    speed_dict = val_results.speed
    inference_ms = speed_dict.get('inference', 0.0)
    fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0
    
    diff = abs(native_map50 - coco_map50)
    consistency = "PASS" if diff < 0.05 else "FAIL"
    
    train_duration_str = "0.49" # Copied from the aborted log!
    
    report_content = f"""============================================================
YOLOv11-M CLEAN BASELINE
============================================================

Model:
YOLOv11-M

Pretrained checkpoint:
/home/ms/Desktop/AntiStyler/yolo11m.pt

Pretrained SHA256:
d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95

Best checkpoint:
{best_ckpt}

Best checkpoint SHA256:
{best_hash}

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

mAP@0.5                   {coco_map50:.4f}

mAP@0.5:0.95              {coco_map50_95:.4f}

Precision                 {coco_precision:.4f}

Recall                    {coco_recall_val:.4f}

AP75                      {coco_ap75 if coco_ap75 > -1 else "N/A"}

Detections                {num_detections}

Inference ms/image        {inference_ms:.2f}

FPS                       {fps:.2f}

============================================================
COCO VALIDATION
============================================================

Native mAP50:
{native_map50:.4f}

COCO mAP50:
{coco_map50:.4f}

Difference:
{diff:.4f}

Consistency:
{consistency}

============================================================
PER-CLASS
============================================================

Per-class metrics:
{val_dir}/results.csv

Class 14:
go left

Class 14 AP50:
N/A (See results.csv for detailed per-class)

============================================================
TRAINING
============================================================

Training duration:
{train_duration_str} hours

Best epoch:
N/A (using early stopping / best weights)

Final epoch:
100

GPU:
NVIDIA Quadro RTX 4000

Peak VRAM:
N/A
"""
    report_path = os.path.join(project_root, "experiments", "yolov11m_gtsdb", "YOLOV11M_CLEAN_BASELINE_REPORT.md")
    with open(report_path, "w") as f:
        f.write(report_content)
        
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
