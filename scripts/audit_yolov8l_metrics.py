import os
from ultralytics.utils.metrics import DetMetrics
from ultralytics.models.yolo.detect import DetectionValidator
import json

def get_ap75():
    # Load Ultralytics Validator without running inference, just using its metric computation on the existing predictions
    print("Performing evaluation via YOLOv8L to grab accurate metrics...")
    
    # Actually, the simplest way is to just use val() on the YOLO object with the existing weights but save_json=True
    # Wait, Ultralytics can evaluate a JSON file directly via pycocotools. 
    # Let's do a fast inference on the directory and correctly grab AP75.
    
    from ultralytics import YOLO
    model = YOLO("experiments/yolov8l_gtsdb/run/weights/best.pt")
    results = model.val(data='experiments/yolov8l_gtsdb/antistyler/dataset.yaml', imgsz=416, split='test', plots=False, verbose=False)
    
    print(f"mAP50: {results.box.map50}")
    print(f"mAP50-95: {results.box.map}")
    print(f"AP shape: {results.box.ap.shape}")
    
    if len(results.box.ap.shape) == 2 and results.box.ap.shape[1] >= 6:
        print(f"AP75: {results.box.ap[:, 5].mean()}")
    else:
        print("AP75 not found in shape")
        
    with open('runs/detect/val3/predictions.json', 'r') as f:
        preds = json.load(f)
    print(f"Total Detections from JSON: {len(preds)}")

if __name__ == "__main__":
    get_ap75()
