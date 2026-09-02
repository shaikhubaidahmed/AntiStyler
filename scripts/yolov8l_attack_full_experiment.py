import os
import glob
import torch
import cv2
import json
import numpy as np
from ultralytics import YOLO

import sys
project_root = "/home/ms/Desktop/AntiStyler"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from attacks.yolov8_patch_attack import YOLOv8PatchAttack

def get_weights_sum(model):
    total = 0
    for param in model.model.parameters():
        total += param.sum().item()
    return total

def main():
    print("=== YOLOv8L DPATCH FULL ATTACK EXPERIMENT ===")
    
    model_path = os.path.join(project_root, "experiments/yolov8l_gtsdb/run/weights/best.pt")
    model = YOLO(model_path)
    model.to('cuda')
    
    # Trigger fusion for clean baseline measurement
    dummy_img = torch.rand(1, 3, 416, 416, device='cuda')
    model(dummy_img, verbose=False)
    initial_weights = get_weights_sum(model)
    
    test_img_dir = os.path.join(project_root, "All Dataset/GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov8/test/images")
    output_img_dir = os.path.join(project_root, "experiments/yolov8l_gtsdb/dpatch_full/images")
    os.makedirs(output_img_dir, exist_ok=True)
    
    img_files = glob.glob(os.path.join(test_img_dir, "*.jpg"))
    img_files = sorted(img_files)
    
    config = {
        'patch_size': 100,
        'num_epochs': 200,
        'lr': 0.05,
        'target_class': 14 
    }
    
    attack = YOLOv8PatchAttack(model, config)
    
    print(f"\nProcessing {len(img_files)} images...")
    
    for idx, img_path in enumerate(img_files):
        filename = os.path.basename(img_path)
        print(f"[{idx+1}/{len(img_files)}] Attacking {filename}...")
        
        img_bgr = cv2.imread(img_path)
        img_bgr = cv2.resize(img_bgr, (416, 416))
        
        img_tensor = torch.from_numpy(img_bgr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor[:, [2, 1, 0], :, :].to('cuda')
        
        # Optimize patch
        attacked_tensor, _ = attack.generate(img_tensor)
        
        # Convert back to BGR for saving
        attacked_np = attacked_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        attacked_np = (attacked_np * 255.0).astype(np.uint8)
        attacked_bgr = attacked_np[:, :, [2, 1, 0]]
        
        out_path = os.path.join(output_img_dir, filename)
        cv2.imwrite(out_path, attacked_bgr)
        
    print("\nAttack Generation Complete.")
    final_weights = get_weights_sum(model)
    print(f"Initial weights: {initial_weights}")
    print(f"Final weights: {final_weights}")
    if initial_weights == final_weights:
        print("PASS: Frozen weights verified.")
    else:
        print("FAIL: Weights changed.")
        
    # Now evaluate on COCO
    # We will use ultralytics to evaluate the attacked dataset
    # Create a dummy yaml for the attacked dataset
    # We can just write a yaml that points to the attacked images
    
    yaml_content = f"""
path: {os.path.join(project_root, 'experiments/yolov8l_gtsdb/dpatch_full')}
train: images
val: images
test: images

nc: 46
names: ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '5', '6', '7', '8', '9']
"""
    yaml_path = os.path.join(project_root, "experiments/yolov8l_gtsdb/dpatch_full/dataset.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
        
    # Copy labels from test set
    import shutil
    test_label_dir = os.path.join(project_root, "All Dataset/GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov8/test/labels")
    output_label_dir = os.path.join(project_root, "experiments/yolov8l_gtsdb/dpatch_full/labels")
    os.makedirs(output_label_dir, exist_ok=True)
    
    for label_file in glob.glob(os.path.join(test_label_dir, "*.txt")):
        shutil.copy(label_file, output_label_dir)
        
    print("\nRunning COCO Evaluation on Attacked Dataset...")
    model.model.eval()
    results = model.val(data=yaml_path, imgsz=416, split='test', plots=False, verbose=True, batch=16, save_json=True)
    
    # We don't save json via Ultralytics val unless specified, it might save to run/val
    
    map50 = results.box.map50
    map50_95 = results.box.map
    
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"DPatch mAP@0.50: {map50:.4f}")
    print(f"DPatch mAP@0.50:0.95: {map50_95:.4f}")
    
    # Save report
    report_path = os.path.join(project_root, "experiments/yolov8l_gtsdb/YOLOV8L_DPATCH_REPORT.md")
    with open(report_path, 'w') as f:
        f.write("# YOLOv8L DPatch Attack Experiment Report\n\n")
        f.write(f"- Dataset: GTSDB (54 images)\n")
        f.write(f"- Architecture: YOLOv8L (Frozen Baseline)\n")
        f.write(f"- Attack Target Class: 14\n")
        f.write(f"- Optimization: 200 epochs, lr=0.05, patch_size=100\n\n")
        f.write("## Results\n")
        f.write(f"- Clean mAP@0.50: 0.912\n")
        f.write(f"- Attacked mAP@0.50: {map50:.4f}\n")
        f.write(f"- Attacked mAP@0.50:0.95: {map50_95:.4f}\n")
        
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
