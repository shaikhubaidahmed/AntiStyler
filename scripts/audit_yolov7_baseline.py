import os
import glob
import pandas as pd
import numpy as np
import json
import hashlib

def get_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def analyze_convergence(results_txt):
    if not os.path.exists(results_txt):
        return "MISSING_FILE"
    
    # YOLOv7 results.txt has columns:
    # 0/1: Epoch/GPU_mem, 2/3/4/5: box_loss, obj_loss, cls_loss, total_loss, 
    # 6: targets, 7: img_size, 8: P, 9: R, 10: mAP@.5, 11: mAP@.5:.95, 
    # 12/13/14: val_box_loss, val_obj_loss, val_cls_loss
    
    with open(results_txt, 'r') as f:
        lines = f.readlines()
        
    data = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 15:
            try:
                map50 = float(parts[10])
                map5095 = float(parts[11])
                data.append((map50, map5095))
            except:
                pass
                
    if not data:
        return "NO_DATA"
        
    maps50 = [x[0] for x in data]
    best_epoch = np.argmax(maps50)
    best_map50 = np.max(maps50)
    
    last_10 = maps50[-10:] if len(maps50) >= 10 else maps50
    diff = last_10[-1] - last_10[0]
    
    # check if map50 plateaued
    if diff > 0.05:
        status = "UNDERTRAINED" # Still improving significantly
    elif np.std(last_10) < 0.02:
        status = "CONVERGED" # Plateaued
    else:
        status = "PARTIALLY CONVERGED"
        
    return {
        "best_epoch": int(best_epoch),
        "best_map50": float(best_map50),
        "status": status,
        "total_epochs": len(maps50)
    }

def analyze_dataset(images_dir, labels_dir):
    labels = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    stats = {
        "images": len(glob.glob(os.path.join(images_dir, "*.jpg"))),
        "labels": 0,
        "classes": set(),
        "widths": [],
        "heights": [],
        "areas": [],
        "empty_images": 0
    }
    
    for lbl in labels:
        if os.path.getsize(lbl) == 0:
            stats["empty_images"] += 1
            continue
            
        with open(lbl, 'r') as f:
            lines = f.readlines()
            if len(lines) == 0:
                stats["empty_images"] += 1
                continue
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    c = int(parts[0])
                    w = float(parts[3])
                    h = float(parts[4])
                    
                    stats["labels"] += 1
                    stats["classes"].add(c)
                    stats["widths"].append(w)
                    stats["heights"].append(h)
                    stats["areas"].append(w * h)
                    
    return stats

def main():
    project_root = "/home/ms/Desktop/AntiStyler"
    dataset_dir = os.path.join(project_root, "All Dataset", "GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov7pytorch")
    
    # 1. Convergence
    conv = analyze_convergence(os.path.join(project_root, "experiments/yolov7_gtsdb/run/results.txt"))
    
    # 2. Dataset Stats
    train_stats = analyze_dataset(os.path.join(dataset_dir, "train/images"), os.path.join(dataset_dir, "train/labels"))
    val_stats = analyze_dataset(os.path.join(dataset_dir, "valid/images"), os.path.join(dataset_dir, "valid/labels"))
    test_stats = analyze_dataset(os.path.join(dataset_dir, "test/images"), os.path.join(dataset_dir, "test/labels"))
    
    # 3. Small Object Analysis (using all splits combined)
    all_w = np.array(train_stats["widths"] + val_stats["widths"] + test_stats["widths"])
    all_h = np.array(train_stats["heights"] + val_stats["heights"] + test_stats["heights"])
    
    # Convert to pixels based on 416x416
    px_w = all_w * 416
    px_h = all_h * 416
    
    total_objects = len(px_w)
    
    small_stats = {
        "w_lt_8": float(np.sum(px_w < 8) / total_objects),
        "w_lt_16": float(np.sum(px_w < 16) / total_objects),
        "w_lt_32": float(np.sum(px_w < 32) / total_objects),
        "w_lt_64": float(np.sum(px_w < 64) / total_objects),
        "mean_px_w": float(np.mean(px_w)),
        "median_px_w": float(np.median(px_w)),
        "mean_px_h": float(np.mean(px_h)),
        "median_px_h": float(np.median(px_h))
    }
    
    results = {
        "convergence": conv,
        "train": {"images": train_stats["images"], "labels": train_stats["labels"]},
        "val": {"images": val_stats["images"], "labels": val_stats["labels"]},
        "test": {"images": test_stats["images"], "labels": test_stats["labels"]},
        "small_objects": small_stats,
        "checkpoint_sha256": get_sha256(os.path.join(project_root, "experiments/yolov7_gtsdb/run/weights/best.pt"))
    }
    
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()
