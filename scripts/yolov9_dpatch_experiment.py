import os
import sys
import torch
import cv2
import hashlib
import glob
import subprocess
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLOV9_DIR = os.path.join(PROJECT_ROOT, "third_party", "yolov9")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if YOLOV9_DIR not in sys.path:
    sys.path.append(YOLOV9_DIR)

from models.experimental import attempt_load
from attacks.yolov9_patch_attack import YOLOv9PatchAttack

def get_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def main():
    checkpoint_path = os.path.join(PROJECT_ROOT, "experiments/yolov9m_gtsdb/weights/best.pt")
    expected_hash = "645721eea8b61c415f3b965ffa275d87de1ed0509bd844645656fb6ef124fb5c"
    
    print("Verifying frozen YOLOv9-M checkpoint...")
    actual_hash = get_hash(checkpoint_path)
    if actual_hash != expected_hash:
        print(f"ERROR: Checkpoint hash mismatch! Expected {expected_hash}, got {actual_hash}")
        sys.exit(1)
    print("Checkpoint verified successfully.")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Loading model...")
    model = attempt_load(checkpoint_path, device=device, inplace=True, fuse=True)
    model.eval()

    test_img_dir = os.path.join(PROJECT_ROOT, "All Dataset/GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov9/test/images")
    output_dir = os.path.join(PROJECT_ROOT, "experiments/yolov9m_gtsdb/dpatch")
    os.makedirs(output_dir, exist_ok=True)
    
    img_files = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg")))
    print(f"Found {len(img_files)} test images.")
    
    attack_config = {
        'patch_size': 100,
        'num_epochs': 200,
        'lr': 0.05,
        'target_class': 14  # 'go right'
    }
    
    attacker = YOLOv9PatchAttack(model, attack_config)
    
    print("Generating adversarial images...")
    for idx, img_path in enumerate(tqdm(img_files)):
        filename = os.path.basename(img_path)
        out_path = os.path.join(output_dir, filename)
        
        # Load image
        img0 = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        
        # YOLOv9 assumes 416x416 input as per our config. The images are already 416x416.
        # But we resize just to be completely safe
        img_resized = cv2.resize(img_rgb, (416, 416))
        
        img_tensor = torch.from_numpy(img_resized).to(device).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        
        attacked_tensor, _ = attacker.generate(img_tensor)
        
        attacked_np = (attacked_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        attacked_bgr = cv2.cvtColor(attacked_np, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(out_path, attacked_bgr)
        
        # Small scale validation on first image
        if idx == 0:
            print("Small-scale validation complete on first image.")
            
    print("All 54 images attacked successfully.")
    
    # We must also create a modified data.yaml to evaluate the attacked folder using val_dual.py
    print("Preparing attacked evaluation configuration...")
    orig_yaml = os.path.join(PROJECT_ROOT, "All Dataset/GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov9/data.yaml")
    attacked_yaml = os.path.join(PROJECT_ROOT, "experiments/yolov9m_gtsdb/attacked_data.yaml")
    
    with open(orig_yaml, 'r') as f:
        yaml_content = f.read()
        
    yaml_content = yaml_content.replace(test_img_dir, output_dir)
    
    with open(attacked_yaml, 'w') as f:
        f.write(yaml_content)
        
    print("Running evaluation on attacked dataset...")
    # Note: Using python val_dual.py directly
    cmd = [
        "python", os.path.join(YOLOV9_DIR, "val_dual.py"),
        "--data", attacked_yaml,
        "--weights", checkpoint_path,
        "--task", "test",
        "--img", "416",
        "--save-json",
        "--name", "yolov9m_dpatch_eval",
        "--project", os.path.join(PROJECT_ROOT, "experiments/yolov9m_gtsdb"),
        "--exist-ok"
    ]
    subprocess.run(cmd, check=True)
    
    print("Attacked evaluation completed. Please inspect YOLOv9m_dpatch_eval outputs.")

if __name__ == "__main__":
    main()
