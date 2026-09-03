import os
import sys
import torch
import cv2

PROJECT_ROOT = "/home/ms/Desktop/AntiStyler"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "yolov12"))

from ultralytics import YOLO
from antistyler.antistyler import AntiStyler

def main():
    print("=== YOLOv12 COMPATIBILITY AUDIT ===")
    
    # DPatch Compatibility
    print("\\nChecking DPatch Compatibility...")
    model = YOLO(os.path.join(PROJECT_ROOT, "yolo12m.pt"))
    model_nn = model.model
    model_nn.train()
    
    dummy_input = torch.randn(1, 3, 416, 416)
    try:
        outputs = model_nn(dummy_input)
        print("Model output type:", type(outputs))
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            print(f"Number of output tensors: {len(outputs)}")
            for i, out in enumerate(outputs):
                print(f"Output {i} shape: {out.shape}")
        else:
            print("Output shape:", outputs.shape)
        print("DPatch spatial objective access: PASS")
    except Exception as e:
        print(f"DPatch compatibility ERROR: {e}")
        
    # AntiStyler Compatibility
    print("\\nChecking AntiStyler Compatibility...")
    try:
        defense = AntiStyler(config_path=os.path.join(PROJECT_ROOT, "configs/antistyler.yaml"))
        dummy_img = torch.rand(1, 3, 416, 416)
        out_img = defense.defend(dummy_img)
        print(f"AntiStyler input shape: {dummy_img.shape}")
        print(f"AntiStyler output shape: {out_img.shape}")
        if dummy_img.shape == out_img.shape:
            print("AntiStyler Compatibility: PASS")
        else:
            print("AntiStyler Compatibility: FAIL (shape mismatch)")
    except Exception as e:
        print(f"AntiStyler compatibility ERROR: {e}")
        
    # Check if GPU has enough VRAM for training (mock check)
    print("\\nChecking Training Feasibility...")
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if vram >= 7.0:
        print(f"GPU VRAM ({vram:.2f} GB) is sufficient for batch_size=16 (estimated)")
    else:
        print(f"GPU VRAM ({vram:.2f} GB) is low. May require batch_size=8 or 4")

if __name__ == "__main__":
    main()
