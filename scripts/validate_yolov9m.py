import os
import torch
import hashlib
from ultralytics import YOLO

def get_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def main():
    model_name = "yolov9m.pt"
    print(f"Validating {model_name}...")
    
    if not os.path.exists(model_name):
        print(f"Error: {model_name} not found.")
        return
        
    sha256 = get_hash(model_name)
    print(f"SHA256: {sha256}")
    
    print("Loading model...")
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
        
    print(f"Architecture: {model.model.__class__.__name__}")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model.to(device)
    
    print("Running minimal inference on GTSDB sample...")
    sample_img = "All Dataset/GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov8/test/images/00026_jpg.rf.4c55e8a72c668d5a0c4a948580757cd9.jpg"
    if os.path.exists(sample_img):
        try:
            results = model(sample_img, verbose=False)
            print(f"Inference SUCCESS. Detected {len(results[0].boxes)} objects.")
        except Exception as e:
            print(f"Inference failed: {e}")
    else:
        print("Sample image not found!")
        
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

if __name__ == "__main__":
    main()
