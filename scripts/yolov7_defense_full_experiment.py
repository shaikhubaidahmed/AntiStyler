import os
import glob
import time
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from antistyler.antistyler import AntiStyler

def main():
    attacked_img_dir = os.path.join(PROJECT_ROOT, "experiments/yolov7_gtsdb/dpatch_full/images")
    defended_img_dir = os.path.join(PROJECT_ROOT, "experiments/yolov7_gtsdb/defended_full/images")
    os.makedirs(defended_img_dir, exist_ok=True)
    
    print("Loading AntiStyler...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    defense = AntiStyler(config_path="configs/antistyler.yaml")
    
    img_files = sorted(glob.glob(os.path.join(attacked_img_dir, "*.jpg")))
    
    if len(img_files) != 54:
        print(f"Error: Expected 54 attacked images, found {len(img_files)}")
        sys.exit(1)
        
    print(f"Applying AntiStyler to {len(img_files)} attacked images...")
    
    total_time = 0.0
    for i, img_path in enumerate(img_files):
        print(f"[{i+1}/{len(img_files)}] Defending {os.path.basename(img_path)}...")
        
        # Load image via cv2 to maintain consistency with YOLOv7
        img_np = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor [B, C, H, W], normalized to [0, 1]
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(device)
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = defense.defend(img_tensor, seed=42, debug=True)
            defended_tensor = outputs["defended_image"]
            
        end_time = time.time()
        total_time += (end_time - start_time)
        
        # Convert back to BGR for cv2
        defended_np = (defended_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        defended_bgr = cv2.cvtColor(defended_np, cv2.COLOR_RGB2BGR)
        
        out_path = os.path.join(defended_img_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, defended_bgr)
        
    mean_time_ms = (total_time / len(img_files)) * 1000
    print(f"Done! Mean AntiStyler processing time: {mean_time_ms:.2f} ms/image")
    
if __name__ == "__main__":
    main()
