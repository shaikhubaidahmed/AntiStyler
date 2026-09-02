import os
import sys
import torch
import hashlib
import cv2

# Ensure YOLOv9 repo is accessible
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLOV9_DIR = os.path.join(PROJECT_ROOT, "third_party", "yolov9")
if YOLOV9_DIR not in sys.path:
    sys.path.append(YOLOV9_DIR)

from models.experimental import attempt_load
from utils.general import non_max_suppression

def get_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def main():
    model_name = os.path.join(PROJECT_ROOT, "yolov9-m.pt")
    print(f"Validating {model_name}...")
    
    if not os.path.exists(model_name):
        print(f"Error: {model_name} not found.")
        return
        
    sha256 = get_hash(model_name)
    print(f"SHA256: {sha256}")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print("Loading official YOLOv9-M model...")
    try:
        model = attempt_load(model_name, device=device, inplace=True, fuse=True)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
        
    # Get architecture info
    # Usually attempt_load returns a Model object
    print(f"Architecture: {model.__class__.__name__}")
    
    # Try getting parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameter count: {total_params:,}")
    
    print("Running minimal inference on GTSDB sample...")
    sample_img = os.path.join(PROJECT_ROOT, "All Dataset/GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov8/test/images/00026_jpg.rf.4c55e8a72c668d5a0c4a948580757cd9.jpg")
    
    if os.path.exists(sample_img):
        try:
            img0 = cv2.imread(sample_img)  # BGR
            # YOLOv9 official code expects RGB, [1, 3, H, W] in 0-1
            img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
            # Just resize to 416 for a quick test if needed, or 640
            img = cv2.resize(img, (640, 640))
            img = torch.from_numpy(img).to(device)
            img = img.float() / 255.0
            img = img.permute(2, 0, 1).unsqueeze(0)
            
            with torch.no_grad():
                pred = model(img, augment=False, visualize=False)
                # Apply NMS
                if isinstance(pred, tuple):
                    pred = pred[0]
                pred = non_max_suppression(pred, 0.25, 0.45, classes=None, max_det=1000)
                
            print(f"Inference SUCCESS. Detected {len(pred[0])} objects.")
        except Exception as e:
            print(f"Inference failed: {e}")
    else:
        print("Sample image not found!")
        
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

if __name__ == "__main__":
    main()
