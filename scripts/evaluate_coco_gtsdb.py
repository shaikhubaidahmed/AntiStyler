import os
import json
import glob
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def convert_yolo_to_coco():
    # Paths
    project_root = "/home/ms/Desktop/AntiStyler"
    dataset_dir = os.path.join(project_root, "All Dataset", "GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov7pytorch")
    test_images_dir = os.path.join(dataset_dir, "test", "images")
    test_labels_dir = os.path.join(dataset_dir, "test", "labels")
    
    output_dir = os.path.join(project_root, "evaluation_data", "gtsdb_coco")
    os.makedirs(output_dir, exist_ok=True)
    
    gt_json_path = os.path.join(output_dir, "instances_gtsdb_test.json")
    
    # Read class names (if available from data.yaml, but we know there are 46 classes, 0 to 45)
    categories = [{"id": i, "name": str(i)} for i in range(46)]
    
    images = []
    annotations = []
    image_name_to_id = {}
    
    image_files = sorted(glob.glob(os.path.join(test_images_dir, "*.jpg")))
    
    ann_id = 1
    for img_id, img_path in enumerate(image_files, start=1):
        filename = os.path.basename(img_path)
        image_name_to_id[filename] = img_id
        # YOLOv7 uses stem as image_id in its raw string output
        stem = os.path.splitext(filename)[0]
        image_name_to_id[stem] = img_id
        
        with Image.open(img_path) as img:
            width, height = img.size
            
        images.append({
            "id": img_id,
            "file_name": filename,
            "width": width,
            "height": height
        })
        
        label_path = os.path.join(test_labels_dir, stem + ".txt")
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * width
                        y_center = float(parts[2]) * height
                        box_width = float(parts[3]) * width
                        box_height = float(parts[4]) * height
                        
                        x_min = max(0, x_center - box_width / 2)
                        y_min = max(0, y_center - box_height / 2)
                        
                        # Clip to image boundaries
                        x_min = min(x_min, width - 1)
                        y_min = min(y_min, height - 1)
                        box_width = min(box_width, width - x_min)
                        box_height = min(box_height, height - y_min)
                        
                        annotations.append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": class_id,
                            "bbox": [x_min, y_min, box_width, box_height],
                            "area": box_width * box_height,
                            "iscrowd": 0
                        })
                        ann_id += 1
                        
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    with open(gt_json_path, 'w') as f:
        json.dump(coco_format, f)
        
    return gt_json_path, image_name_to_id, len(images), len(annotations)

def fix_predictions_json(image_name_to_id):
    project_root = "/home/ms/Desktop/AntiStyler"
    preds_path = os.path.join(project_root, "experiments", "yolov7_gtsdb", "eval_clean", "best_predictions.json")
    fixed_preds_path = os.path.join(project_root, "evaluation_data", "gtsdb_coco", "best_predictions_fixed.json")
    
    with open(preds_path, 'r') as f:
        preds = json.load(f)
        
    fixed_preds = []
    for p in preds:
        img_str_id = p["image_id"]
        if img_str_id in image_name_to_id:
            p["image_id"] = image_name_to_id[img_str_id]
            fixed_preds.append(p)
            
    with open(fixed_preds_path, 'w') as f:
        json.dump(fixed_preds, f)
        
    return fixed_preds_path, len(fixed_preds)

def evaluate_coco(gt_json_path, preds_json_path):
    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(preds_json_path)
    
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    return coco_eval.stats

def main():
    print("Generating COCO GT JSON...")
    gt_json_path, image_name_to_id, num_imgs, num_anns = convert_yolo_to_coco()
    print(f"Generated GT JSON with {num_imgs} images and {num_anns} annotations.")
    
    print("Fixing predictions JSON (mapping string IDs to int)...")
    fixed_preds_path, num_preds = fix_predictions_json(image_name_to_id)
    print(f"Mapped {num_preds} predictions.")
    
    print("Running COCOeval...")
    stats = evaluate_coco(gt_json_path, fixed_preds_path)
    
    mAP_50_95 = stats[0]
    mAP_50 = stats[1]
    
    print(f"COCO mAP@0.50: {mAP_50}")
    print(f"COCO mAP@0.50:0.95: {mAP_50_95}")
    
    project_root = "/home/ms/Desktop/AntiStyler"
    results_dir = os.path.join(project_root, "experiments", "yolov7_gtsdb", "results")
    coco_metrics_path = os.path.join(results_dir, "coco_metrics.json")
    
    coco_metrics = {
        "mAP_0.50": float(mAP_50),
        "mAP_0.50_0.95": float(mAP_50_95),
        "test_images": num_imgs,
        "gt_annotations": num_anns,
        "predictions": num_preds
    }
    
    with open(coco_metrics_path, 'w') as f:
        json.dump(coco_metrics, f, indent=4)
        
    print(f"Saved COCO metrics to {coco_metrics_path}")

if __name__ == "__main__":
    main()
