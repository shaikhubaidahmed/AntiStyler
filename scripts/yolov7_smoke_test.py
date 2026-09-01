"""
YOLOv7 Smoke Test on GTSDB
Validates: model loading, inference, prediction format, visual output.
NOT a research evaluation.
"""
import sys
import os
import glob
import cv2
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors.yolov7_detector import YOLOv7Detector

YOLOV7_WEIGHTS = "third_party/yolov7/yolov7.pt"
DATASET_ROOT = "All Dataset/GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov7pytorch"
OUTPUT_DIR = "debug_outputs/yolov7_smoke_test"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load model
    print("Loading YOLOv7...")
    detector = YOLOv7Detector()
    detector.load_model(YOLOV7_WEIGHTS, img_size=416)
    
    # 2. Load a few test images
    test_images = sorted(glob.glob(os.path.join(DATASET_ROOT, "test", "images", "*.jpg")))[:5]
    if not test_images:
        print("ERROR: No test images found.")
        sys.exit(1)
    
    print(f"Found {len(test_images)} test images for smoke test.")
    
    all_pass = True
    for img_path in test_images:
        print(f"\nProcessing: {os.path.basename(img_path)}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ERROR: Could not read image")
            all_pass = False
            continue
        
        # 3. Run inference
        preds = detector.get_predictions(img, conf_threshold=0.25, iou_threshold=0.45)
        print(f"  Detections: {len(preds)}")
        
        # 4. Validate predictions
        for p in preds:
            bbox = p["bbox"]
            conf = p["confidence"]
            cls_id = p["class_id"]
            cls_name = p["class_name"]
            
            # Check for NaN/Inf
            for v in bbox:
                if np.isnan(v) or np.isinf(v):
                    print(f"  ERROR: NaN/Inf in bbox: {bbox}")
                    all_pass = False
            if np.isnan(conf) or np.isinf(conf):
                print(f"  ERROR: NaN/Inf in confidence: {conf}")
                all_pass = False
            
            # Check confidence range
            if not (0.0 <= conf <= 1.0):
                print(f"  ERROR: confidence {conf} out of [0, 1]")
                all_pass = False
            
            # Check class ID validity
            if cls_id < 0 or cls_id >= 80:  # COCO pretrained has 80 classes
                print(f"  ERROR: class_id {cls_id} out of range")
                all_pass = False
            
            # Check bounding box validity
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                print(f"  ERROR: invalid bbox {bbox}")
                all_pass = False
                
            print(f"  [{cls_name}] conf={conf:.3f} bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
        
        # 5. Save visual output
        img_vis = img.copy()
        for p in preds:
            x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
            conf = p["confidence"]
            cls_name = p["class_name"]
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_vis, f"{cls_name} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        out_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
        cv2.imwrite(out_path, img_vis)
        print(f"  Saved: {out_path}")
    
    # 6. Measure inference time
    print("\nMeasuring inference time...")
    img = cv2.imread(test_images[0])
    avg_time = detector.get_inference_time(img, num_runs=50)
    print(f"Average inference time: {avg_time:.2f} ms ({1000/avg_time:.1f} FPS)")
    
    # 7. Summary
    print("\n" + "="*50)
    if all_pass:
        print("SMOKE TEST: PASS")
    else:
        print("SMOKE TEST: FAIL (see errors above)")
    print("="*50)

if __name__ == "__main__":
    main()
