import os
import glob
from ultralytics import YOLO

def main():
    project_root = "/home/ms/Desktop/AntiStyler"
    model_path = os.path.join(project_root, "yolov8s.pt")
    
    print("1. Loading Model...")
    model = YOLO(model_path)
    print("Model loaded successfully.")
    print(f"Pretrained classes: {len(model.names)}")
    
    print("\n2. Smoke Test Inference...")
    test_img_dir = os.path.join(project_root, "All Dataset/GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov8/test/images")
    img_files = glob.glob(os.path.join(test_img_dir, "*.jpg"))[:5]
    
    for img_path in img_files:
        results = model(img_path, device=0, verbose=False)
        print(f"Inference on {os.path.basename(img_path)}: {len(results[0].boxes)} detections")
        
    print("\n3. Short Training Sanity Test (1 epoch)...")
    data_yaml = os.path.join(project_root, "configs/datasets/gtsdb_yolov8.yaml")
    
    # Run 1 epoch on GTSDB
    results = model.train(
        data=data_yaml,
        epochs=1,
        imgsz=416,
        batch=16,
        device=0,
        project=os.path.join(project_root, "experiments", "yolov8_gtsdb"),
        name="sanity_check",
        exist_ok=True,
        seed=42
    )
    
    print("\nSanity check completed successfully.")

if __name__ == "__main__":
    main()
