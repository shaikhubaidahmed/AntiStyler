import os
import glob
import torch
import cv2
import hashlib
from ultralytics import YOLO

# Add attacks to path
import sys
project_root = "/home/ms/Desktop/AntiStyler"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from attacks.yolov8_patch_attack import YOLOv8PatchAttack

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
    model_path = os.path.join(project_root, "experiments/yolov8l_gtsdb/run/weights/best.pt")
    
    print(f"1. Checking Frozen Checkpoint Hash...")
    initial_hash = get_hash(model_path)
    print(f"Initial Hash: {initial_hash}")
    if initial_hash != "c9554d26f09f377a048c32ff7ade71baad8e5b90ad678fc43f32a1df74fbaddc":
        print("FAIL: Checkpoint hash does not match!")
        return
        
    print(f"\n2. Loading Model...")
    model = YOLO(model_path)
    model.to('cuda')
    
    # Do a dummy forward pass to trigger Ultralytics layer fusion
    dummy_img = torch.rand(1, 3, 416, 416, device='cuda')
    model(dummy_img, verbose=False)
    
    initial_weight_sum = get_weights_sum(model)
    
    # Target Class 14: go right (same as YOLOv7 experiment config: 14) 
    # Let's use target_class = 14 to be consistent.
    config = {
        'patch_size': 100,
        'num_epochs': 200, # Required by prompt
        'lr': 0.05,
        'target_class': 14 
    }
    attack = YOLOv8PatchAttack(model, config)
    
    print("\n3. Running Attack Validation...")
    test_img_dir = os.path.join(project_root, "All Dataset/GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov8/test/images")
    img_files = glob.glob(os.path.join(test_img_dir, "*.jpg"))[:3]
    
    for img_path in img_files:
        print(f"\nTesting {os.path.basename(img_path)}...")
        
        # Load and prep image
        img_bgr = cv2.imread(img_path)
        img_bgr = cv2.resize(img_bgr, (416, 416))
        
        img_tensor = torch.from_numpy(img_bgr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        # RGB
        img_tensor = img_tensor[:, [2, 1, 0], :, :].to('cuda')
        
        # Clean prediction
        model.model.eval() # Must set back to eval for normal YOLO prediction
        clean_results = model(img_tensor, verbose=False)
        clean_preds = clean_results[0].boxes
        print(f"  Clean detections: {len(clean_preds)}")
        for box in clean_preds:
            print(f"    - Class {int(box.cls[0].item())}: Conf {box.conf[0].item():.2f}")
            
        # Generate attack
        print(f"  Optimizing patch for 200 epochs...")
        attacked_tensor, patch = attack.generate(img_tensor)
        
        # Attacked prediction
        model.model.eval()
        attacked_results = model(attacked_tensor, verbose=False)
        attacked_preds = attacked_results[0].boxes
        print(f"  Attacked detections: {len(attacked_preds)}")
        for box in attacked_preds:
            print(f"    - Class {int(box.cls[0].item())}: Conf {box.conf[0].item():.2f}")
            
    print("\n4. Verifying Frozen Integrity...")
    final_weight_sum = get_weights_sum(model)
    print(f"  Initial weight sum: {initial_weight_sum}")
    print(f"  Final weight sum: {final_weight_sum}")
    if abs(initial_weight_sum - final_weight_sum) < 1e-4:
        print("  PASS: Detector weights remain unchanged.")
    else:
        print("  FAIL: Detector weights CHANGED during attack optimization!")

if __name__ == "__main__":
    main()
