import os
import glob
import json
import hashlib
import numpy as np

def get_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def check_images(dir_path):
    return set([os.path.basename(p) for p in glob.glob(os.path.join(dir_path, "*.jpg"))])

def check_labels(dir_path):
    return set([os.path.basename(p) for p in glob.glob(os.path.join(dir_path, "*.txt"))])

def main():
    project_root = "/home/ms/Desktop/AntiStyler"
    clean_img_dir = os.path.join(project_root, "All Dataset/GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov7pytorch/test/images")
    clean_lbl_dir = os.path.join(project_root, "All Dataset/GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov7pytorch/test/labels")
    
    attacked_img_dir = os.path.join(project_root, "experiments/yolov7_gtsdb/dpatch_full/images")
    attacked_lbl_dir = os.path.join(project_root, "experiments/yolov7_gtsdb/dpatch_full/labels")
    
    defended_img_dir = os.path.join(project_root, "experiments/yolov7_gtsdb/defended_full/images")
    defended_lbl_dir = os.path.join(project_root, "experiments/yolov7_gtsdb/defended_full/labels")
    
    clean_imgs = check_images(clean_img_dir)
    attacked_imgs = check_images(attacked_img_dir)
    defended_imgs = check_images(defended_img_dir)
    
    clean_lbls = check_labels(clean_lbl_dir)
    attacked_lbls = check_labels(attacked_lbl_dir)
    defended_lbls = check_labels(defended_lbl_dir)
    
    image_correspondence = (clean_imgs == attacked_imgs) and (attacked_imgs == defended_imgs) and len(clean_imgs) == 54
    gt_correspondence = (clean_lbls == attacked_lbls) and (attacked_lbls == defended_lbls) and len(clean_lbls) == 54
    
    # Recalculate
    c_mAP50 = 0.316
    a_mAP50 = 0.077
    d_mAP50 = 0.234
    
    recalc_deg = c_mAP50 - a_mAP50
    recalc_rec = d_mAP50 - a_mAP50
    recalc_rec_pct = (recalc_rec / recalc_deg) * 100 if recalc_deg != 0 else 0
    recalc_rel_clean = (d_mAP50 / c_mAP50) * 100
    
    c_mAP50_95 = 0.201
    a_mAP50_95 = 0.046
    d_mAP50_95 = 0.147
    
    recalc_rec_pct_95 = ((d_mAP50_95 - a_mAP50_95) / (c_mAP50_95 - a_mAP50_95)) * 100
    
    results = {
        "images": {
            "clean_count": len(clean_imgs),
            "attacked_count": len(attacked_imgs),
            "defended_count": len(defended_imgs),
            "correspondence_pass": image_correspondence
        },
        "labels": {
            "clean_count": len(clean_lbls),
            "attacked_count": len(attacked_lbls),
            "defended_count": len(defended_lbls),
            "correspondence_pass": gt_correspondence
        },
        "recalculation": {
            "reported_recovery": 65.69,
            "recalculated_recovery": round(recalc_rec_pct, 2),
            "mAP50_95_recovery": round(recalc_rec_pct_95, 2),
            "pass": abs(65.69 - recalc_rec_pct) < 0.05
        }
    }
    
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()
