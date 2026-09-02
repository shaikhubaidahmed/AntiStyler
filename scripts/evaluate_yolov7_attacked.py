"""
Evaluate YOLOv7 on GTSDB ATTACKED test split using official test.py
"""
import sys
import os
import json
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_DIR = os.path.join(PROJECT_ROOT, "third_party", "yolov7")
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiments", "yolov7_gtsdb")
WEIGHTS = os.path.join(EXPERIMENT_DIR, "run", "weights", "best.pt")
DATA_YAML = os.path.join(PROJECT_ROOT, "configs", "datasets", "gtsdb_yolov7_attacked.yaml")
OUTPUT_DIR = os.path.join(EXPERIMENT_DIR, "dpatch_full")

from evaluate_coco_gtsdb import evaluate_coco

def main():
    print("Running YOLOv7 evaluation on attacked images (test.py)...")
    cmd = [
        "python", "test.py",
        "--weights", WEIGHTS,
        "--data", DATA_YAML,
        "--task", "test",
        "--batch-size", "16",
        "--img-size", "416",
        "--conf-thres", "0.001",
        "--iou-thres", "0.65",
        "--device", "0",
        "--save-json",
        "--project", EXPERIMENT_DIR,
        "--name", "eval_attacked",
        "--exist-ok"
    ]
    
    result = subprocess.run(cmd, cwd=YOLO_DIR, capture_output=True, text=True)
    print(result.stdout)
    
    yolov7_preds_json = os.path.join(EXPERIMENT_DIR, "eval_attacked", "best_predictions.json")
    
    # Run COCO evaluation
    gt_json = os.path.join(PROJECT_ROOT, "evaluation_data", "gtsdb_coco", "instances_gtsdb_test.json")
    
    # Need to convert string IDs to int IDs
    with open(gt_json, 'r') as f:
        coco_gt = json.load(f)
        
    image_name_to_id = {}
    for img in coco_gt['images']:
        filename = img['file_name']
        stem = os.path.splitext(filename)[0]
        image_name_to_id[filename] = img['id']
        image_name_to_id[stem] = img['id']
        
    with open(yolov7_preds_json, 'r') as f:
        preds = json.load(f)
        
    fixed_preds = []
    for p in preds:
        img_str_id = p["image_id"]
        if img_str_id in image_name_to_id:
            p["image_id"] = image_name_to_id[img_str_id]
            fixed_preds.append(p)
            
    fixed_preds_path = os.path.join(EXPERIMENT_DIR, "eval_attacked", "best_predictions_fixed.json")
    with open(fixed_preds_path, 'w') as f:
        json.dump(fixed_preds, f)
    
    print("\nRunning COCO Evaluation on Attacked Predictions...")
    stats = evaluate_coco(gt_json, fixed_preds_path)
    
    print(f"\nAttacked COCO mAP@0.50: {stats[1]}")

if __name__ == "__main__":
    main()
