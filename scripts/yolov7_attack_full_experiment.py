import os
import sys
import json
import glob
import time
import torch
import cv2
import numpy as np
from PIL import Image
import shutil

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "yolov7"))
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

from attacks.yolov7_patch_attack import YOLOv7PatchAttack
from scripts.evaluate_coco_gtsdb import evaluate_coco

def load_image(img_path, img_size=416):
    img0 = cv2.imread(img_path)
    img = letterbox(img0, img_size, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor, img0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = "/home/ms/Desktop/AntiStyler"
    weights_path = "experiments/yolov7_gtsdb/run/weights/best.pt"
    test_images_dir = os.path.join(project_root, "All Dataset", "GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov7pytorch", "test", "images")
    output_dir = os.path.join(project_root, "experiments", "yolov7_gtsdb", "dpatch_full")
    attacked_img_dir = os.path.join(output_dir, "images")
    vis_dir = os.path.join(output_dir, "visualizations")
    
    os.makedirs(attacked_img_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"Loading YOLOv7 model from {weights_path}...")
    model = attempt_load(weights_path, map_location=device)
    model.eval()
    
    attack_config = {
        'patch_size': 100,
        'num_epochs': 200,
        'lr': 0.1,
        'target_class': 0 # ANIMALS as dummy class
    }
    
    attack = YOLOv7PatchAttack(model, attack_config)
    
    test_files = sorted(glob.glob(os.path.join(test_images_dir, "*.jpg")))
    
    gt_json_path = os.path.join(project_root, "evaluation_data", "gtsdb_coco", "instances_gtsdb_test.json")
    with open(gt_json_path, 'r') as f:
        coco_gt = json.load(f)
        
    img_name_to_id = {img['file_name']: img['id'] for img in coco_gt['images']}
    
    clean_preds_json = []
    attacked_preds_json = []
    
    total_opt_time = 0
    total_clean_inf_time = 0
    total_attacked_inf_time = 0
    
    num_clean_dets = 0
    num_attacked_dets = 0
    
    for i, f in enumerate(test_files):
        img_tensor, img0 = load_image(f)
        img_tensor = img_tensor.to(device)
        img_id = img_name_to_id[os.path.basename(f)]
        
        print(f"[{i+1}/{len(test_files)}] Attacking {os.path.basename(f)}...")
        
        t0 = time.time()
        attacked_tensor, _ = attack.generate(img_tensor)
        t_opt = time.time() - t0
        total_opt_time += t_opt
        
        # Save attacked image
        attacked_img_np = attacked_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        attacked_img_np = attacked_img_np.astype(np.uint8)
        attacked_img_np = cv2.cvtColor(attacked_img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(attacked_img_dir, os.path.basename(f)), attacked_img_np)
        
        # Inference
        with torch.no_grad():
            t1 = time.time()
            c_pred = model(img_tensor, augment=False)[0]
            t2 = time.time()
            c_pred = non_max_suppression(c_pred, 0.001, 0.65)[0]
            if len(c_pred):
                c_pred[:, :4] = scale_coords(img_tensor.shape[2:], c_pred[:, :4], img0.shape).round()
            t3 = time.time()
            total_clean_inf_time += (t3 - t1)
            
            t4 = time.time()
            a_pred = model(attacked_tensor, augment=False)[0]
            t5 = time.time()
            a_pred = non_max_suppression(a_pred, 0.001, 0.65)[0]
            if len(a_pred):
                a_pred[:, :4] = scale_coords(img_tensor.shape[2:], a_pred[:, :4], img0.shape).round()
            t6 = time.time()
            total_attacked_inf_time += (t6 - t4)
            
        num_clean_dets += len(c_pred)
        num_attacked_dets += len(a_pred)
            
        def format_pred(pred, img_id):
            res = []
            for det in pred:
                x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                w = x2 - x1
                h = y2 - y1
                res.append({
                    "image_id": img_id,
                    "category_id": int(cls),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(conf)
                })
            return res
            
        clean_preds_json.extend(format_pred(c_pred, img_id))
        attacked_preds_json.extend(format_pred(a_pred, img_id))
        
    clean_json_path = os.path.join(output_dir, "clean_preds_full.json")
    attacked_json_path = os.path.join(output_dir, "attacked_preds_full.json")
    
    with open(clean_json_path, 'w') as f:
        json.dump(clean_preds_json, f)
    with open(attacked_json_path, 'w') as f:
        json.dump(attacked_preds_json, f)
        
    print("\nEvaluating Full Clean Set...")
    clean_stats = evaluate_coco(gt_json_path, clean_json_path)
    clean_mAP50_95 = clean_stats[0]
    clean_mAP50 = clean_stats[1]
    clean_ap75 = clean_stats[2]
    clean_recall = clean_stats[8]
    
    print("\nEvaluating Full Attacked Set...")
    attacked_stats = evaluate_coco(gt_json_path, attacked_json_path)
    attacked_mAP50_95 = attacked_stats[0]
    attacked_mAP50 = attacked_stats[1]
    attacked_ap75 = attacked_stats[2]
    attacked_recall = attacked_stats[8]
    
    results = {
        "clean": {
            "mAP50": clean_mAP50,
            "mAP50_95": clean_mAP50_95,
            "AP75": clean_ap75,
            "recall": clean_recall,
            "detections": num_clean_dets,
            "mean_inference_time": total_clean_inf_time / len(test_files)
        },
        "attacked": {
            "mAP50": attacked_mAP50,
            "mAP50_95": attacked_mAP50_95,
            "AP75": attacked_ap75,
            "recall": attacked_recall,
            "detections": num_attacked_dets,
            "mean_inference_time": total_attacked_inf_time / len(test_files)
        },
        "mean_optimization_time": total_opt_time / len(test_files),
        "total_images": len(test_files)
    }
    
    with open(os.path.join(output_dir, "YOLOV7_FULL_DPATTACK_RESULTS.json"), 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\nDONE.")

if __name__ == "__main__":
    main()
