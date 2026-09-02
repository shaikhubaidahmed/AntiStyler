import os
import glob
import time
import cv2
import torch
import hashlib
import numpy as np
from ultralytics import YOLO

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from antistyler.antistyler import AntiStyler

def get_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def get_weights_sum(model):
    total = 0
    for param in model.model.parameters():
        total += param.sum().item()
    return total

def main():
    print("=== YOLOv8L ANTISTYLER DEFENSE SMALL-SCALE VALIDATION ===")
    
    yolo_ckpt = os.path.join(PROJECT_ROOT, "experiments/yolov8l_gtsdb/run/weights/best.pt")
    ckpt_hash = get_hash(yolo_ckpt)
    print(f"YOLOv8L Checkpoint SHA256: {ckpt_hash}")
    
    if ckpt_hash != "c9554d26f09f377a048c32ff7ade71baad8e5b90ad678fc43f32a1df74fbaddc":
        print("FAIL: Checkpoint hash does not match!")
        return
        
    print("\nLoading YOLOv8L (Frozen Baseline)...")
    model = YOLO(yolo_ckpt)
    model.to('cuda')
    # Trigger fusion for clean baseline measurement
    dummy_img = torch.rand(1, 3, 416, 416, device='cuda')
    model(dummy_img, verbose=False)
    initial_weights = get_weights_sum(model)
    
    vgg_ckpt_path = "/home/ms/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth"
    if os.path.exists(vgg_ckpt_path):
        vgg_hash = get_hash(vgg_ckpt_path)
        print(f"\nVGG19 Checkpoint found at {vgg_ckpt_path}")
        print(f"VGG19 SHA256: {vgg_hash}")
    else:
        print(f"\nVGG19 Checkpoint NOT FOUND at {vgg_ckpt_path}! (Assuming torchvision download)")
        
    print("\nLoading AntiStyler...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    defense = AntiStyler(config_path="configs/antistyler.yaml")
    
    attacked_img_dir = os.path.join(PROJECT_ROOT, "experiments/yolov8l_gtsdb/dpatch_full/images")
    img_files = sorted(glob.glob(os.path.join(attacked_img_dir, "*.jpg")))[:3]
    
    print("\nRunning Small-Scale Defense Validation...")
    for i, img_path in enumerate(img_files):
        print(f"\nProcessing {os.path.basename(img_path)}...")
        
        # Load image
        img_np = cv2.imread(img_path)
        if img_np is None:
            print("  FAIL: Image failed to load.")
            continue
        
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(device)
        
        print("  Attacked image shape:", img_tensor.shape)
        
        # Apply AntiStyler
        with torch.no_grad(): # AntiStyler uses its own internal optimizations, but the outer wrapper uses no_grad
            outputs = defense.defend(img_tensor, seed=42, debug=True)
            defended_tensor = outputs["defended_image"]
            mask_tensor = outputs["final_mask"]
            
        print("  Defended image shape:", defended_tensor.shape)
        
        # Validation checks
        if torch.isnan(defended_tensor).any() or torch.isinf(defended_tensor).any():
            print("  FAIL: Defended image contains NaN/Inf!")
        else:
            print("  PASS: No NaN/Inf detected.")
            
        diff = torch.abs(img_tensor.cpu() - defended_tensor).sum().item()
        print(f"  Difference between attacked and defended: {diff:.2f}")
        
        if mask_tensor.sum().item() > 0:
            print("  PASS: Mask is non-empty.")
            if diff > 0:
                print("  PASS: Defended image is different from attacked image.")
            else:
                print("  FAIL: Defended image is IDENTICAL to attacked image despite non-empty mask!")
        else:
            print("  INFO: Mask is empty (no patch detected).")
            
        # Convert to BGR format matching YOLO inference expected format
        # Normally ultralytics model(img_np) handles BGR natively
        defended_np = (defended_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        defended_bgr = cv2.cvtColor(defended_np, cv2.COLOR_RGB2BGR)
        
        # Verify YOLOv8 inference
        print("  Running YOLOv8L inference on attacked image...")
        model.model.eval()
        att_results = model(img_np, verbose=False)
        att_preds = att_results[0].boxes
        print(f"  Attacked Detections: {len(att_preds)}")
        for box in att_preds:
            if box.conf[0].item() > 0.5:
                print(f"    - Class {int(box.cls[0].item())}: Conf {box.conf[0].item():.2f}")
            
        print("  Running YOLOv8L inference on defended image...")
        def_results = model(defended_bgr, verbose=False)
        def_preds = def_results[0].boxes
        print(f"  Defended Detections: {len(def_preds)}")
        for box in def_preds:
            if box.conf[0].item() > 0.5:
                print(f"    - Class {int(box.cls[0].item())}: Conf {box.conf[0].item():.2f}")
                
    final_weights = get_weights_sum(model)
    print("\nIntegrity Verification:")
    if initial_weights == final_weights:
        print("PASS: YOLOv8L weights remain unchanged.")
    else:
        print("FAIL: YOLOv8L weights CHANGED!")

if __name__ == "__main__":
    main()
