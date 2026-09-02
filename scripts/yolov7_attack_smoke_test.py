import os
import sys
import json
import glob
import time
import torch
import cv2
import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "yolov7"))
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

from attacks.yolov7_patch_attack import YOLOv7PatchAttack
from scripts.evaluate_coco_gtsdb import evaluate_coco

def load_image(img_path, img_size=416):
    img0 = cv2.imread(img_path)
    img = letterbox(img0, img_size, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor, img0

def save_visualization(img_tensor, boxes, save_path):
    img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
    img = img.astype(np.uint8).copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    cv2.imwrite(save_path, img)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = "/home/ms/Desktop/AntiStyler"
    weights_path = "experiments/yolov7_gtsdb/run/weights/best.pt"
    test_images_dir = os.path.join(project_root, "All Dataset", "GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov7pytorch", "test", "images")
    output_dir = os.path.join(project_root, "experiments", "yolov7_gtsdb", "attacks")
    vis_dir = os.path.join(output_dir, "visualizations")
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
    
    # SINGLE IMAGE TEST
    print("\n--- SINGLE IMAGE VALIDATION ---")
    img_path = test_files[0]
    img_tensor, img0 = load_image(img_path)
    img_tensor = img_tensor.to(device)
    
    print("Running attack on single image...")
    t0 = time.time()
    attacked_tensor, patch = attack.generate(img_tensor)
    t_opt = time.time() - t0
    
    # Check NaN/Inf
    assert not torch.isnan(attacked_tensor).any(), "NaN found in attacked image!"
    assert not torch.isinf(attacked_tensor).any(), "Inf found in attacked image!"
    
    # Save visualizations
    clean_vis_path = os.path.join(vis_dir, "single_clean.jpg")
    attacked_vis_path = os.path.join(vis_dir, "single_attacked.jpg")
    
    with torch.no_grad():
        t1 = time.time()
        clean_pred = model(img_tensor, augment=False)[0]
        t_inf = time.time() - t1
        clean_pred = non_max_suppression(clean_pred, 0.001, 0.65)[0]
        
        attacked_pred = model(attacked_tensor, augment=False)[0]
        attacked_pred = non_max_suppression(attacked_pred, 0.001, 0.65)[0]
        
    save_visualization(img_tensor, clean_pred, clean_vis_path)
    save_visualization(attacked_tensor, attacked_pred, attacked_vis_path)
    print(f"Single image attack successful. Optimization time: {t_opt:.2f}s, Inference time: {t_inf:.3f}s")
    
    # 5-IMAGE VALIDATION
    print("\n--- 5-IMAGE SUBSET VALIDATION ---")
    subset_files = test_files[:5]
    
    clean_preds_json = []
    attacked_preds_json = []
    
    # Load GT mapping from evaluate_coco_gtsdb.py logic
    gt_json_path = os.path.join(project_root, "evaluation_data", "gtsdb_coco", "instances_gtsdb_test.json")
    with open(gt_json_path, 'r') as f:
        coco_gt = json.load(f)
        
    img_name_to_id = {img['file_name']: img['id'] for img in coco_gt['images']}
    
    # Filter GT for just these 5 images
    subset_img_ids = [img_name_to_id[os.path.basename(f)] for f in subset_files]
    subset_gt = {
        "images": [img for img in coco_gt['images'] if img['id'] in subset_img_ids],
        "annotations": [ann for ann in coco_gt['annotations'] if ann['image_id'] in subset_img_ids],
        "categories": coco_gt['categories']
    }
    subset_gt_path = os.path.join(output_dir, "subset_gt.json")
    with open(subset_gt_path, 'w') as f:
        json.dump(subset_gt, f)
        
    for i, f in enumerate(subset_files):
        img_tensor, img0 = load_image(f)
        img_tensor = img_tensor.to(device)
        img_id = img_name_to_id[os.path.basename(f)]
        
        print(f"Attacking image {i+1}/5...")
        attacked_tensor, _ = attack.generate(img_tensor)
        
        with torch.no_grad():
            c_pred = model(img_tensor, augment=False)[0]
            c_pred = non_max_suppression(c_pred, 0.001, 0.65)[0]
            
            a_pred = model(attacked_tensor, augment=False)[0]
            a_pred = non_max_suppression(a_pred, 0.001, 0.65)[0]
            
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
        
    clean_json_path = os.path.join(output_dir, "subset_clean_preds.json")
    attacked_json_path = os.path.join(output_dir, "subset_attacked_preds.json")
    
    with open(clean_json_path, 'w') as f:
        json.dump(clean_preds_json, f)
    with open(attacked_json_path, 'w') as f:
        json.dump(attacked_preds_json, f)
        
    print("\nEvaluating Clean Subset...")
    clean_stats = evaluate_coco(subset_gt_path, clean_json_path)
    clean_mAP50 = clean_stats[1]
    
    print("\nEvaluating Attacked Subset...")
    attacked_stats = evaluate_coco(subset_gt_path, attacked_json_path)
    attacked_mAP50 = attacked_stats[1]
    
    diff = clean_mAP50 - attacked_mAP50
    print(f"\nRESULTS:")
    print(f"Clean mAP@0.50: {clean_mAP50:.4f}")
    print(f"Attacked mAP@0.50: {attacked_mAP50:.4f}")
    print(f"Difference: {diff:.4f}")
    
    results = {
        "clean_mAP50": clean_mAP50,
        "attacked_mAP50": attacked_mAP50,
        "difference": diff,
        "optimization_time": t_opt,
        "inference_time": t_inf
    }
    
    with open(os.path.join(output_dir, "YOLOV7_ATTACK_VALIDATION.json"), 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\nDONE.")

if __name__ == "__main__":
    main()
