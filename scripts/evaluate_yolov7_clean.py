"""
Evaluate YOLOv7 on GTSDB CLEAN test split.
Outputs results to experiments/yolov7_gtsdb/results/clean_metrics.json
"""
import sys
import os
import json
import subprocess
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_DIR = os.path.join(PROJECT_ROOT, "third_party", "yolov7")
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiments", "yolov7_gtsdb")
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
WEIGHTS = os.path.join(EXPERIMENT_DIR, "run", "weights", "best.pt")
DATA_YAML = os.path.join(PROJECT_ROOT, "configs", "datasets", "gtsdb_yolov7.yaml")

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if not os.path.exists(WEIGHTS):
        print(f"Error: Weights not found at {WEIGHTS}. Have you trained the model?")
        sys.exit(1)
        
    print("Running YOLOv7 evaluation (test.py)...")
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
        "--save-json",  # Save COCO format json
        "--project", EXPERIMENT_DIR,
        "--name", "eval_clean",
        "--exist-ok"
    ]
    
    # Run from YOLO directory
    result = subprocess.run(cmd, cwd=YOLO_DIR, capture_output=True, text=True)
    
    # Parse output to extract metrics
    metrics = {
        "mAP_0.50": None,
        "mAP_0.50_0.95": None,
        "Precision": None,
        "Recall": None,
        "Class_Metrics": {}
    }
    
    output = result.stdout
    print(output)
    
    # Simple parser for YOLOv7 output
    # Output format:
    # Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100%|...
    # all          54          54       0.8       0.9        0.85       0.6
    
    lines = output.split('\n')
    parsing_classes = False
    for line in lines:
        if "Class" in line and "Images" in line and "Labels" in line and "mAP@.5" in line:
            parsing_classes = True
            continue
            
        if parsing_classes:
            parts = line.strip().split()
            if len(parts) >= 7:
                cls_name = parts[0]
                try:
                    p = float(parts[-4])
                    r = float(parts[-3])
                    map50 = float(parts[-2])
                    map5095 = float(parts[-1])
                    
                    if cls_name == "all":
                        metrics["Precision"] = p
                        metrics["Recall"] = r
                        metrics["mAP_0.50"] = map50
                        metrics["mAP_0.50_0.95"] = map5095
                    else:
                        # Depending on class name length, it might be split
                        if len(parts) > 7:
                            cls_name = " ".join(parts[:-6])
                        metrics["Class_Metrics"][cls_name] = {
                            "Precision": p,
                            "Recall": r,
                            "mAP_0.50": map50,
                            "mAP_0.50_0.95": map5095
                        }
                except ValueError:
                    pass
    
    with open(os.path.join(RESULTS_DIR, "clean_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Evaluation complete. mAP@0.50: {metrics['mAP_0.50']}")

if __name__ == "__main__":
    main()
