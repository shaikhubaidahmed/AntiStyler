import os
import gc
import json
import torch
from ultralytics import YOLO

def train_with_fallback():
    project_root = "/home/ms/Desktop/AntiStyler"
    model_path = os.path.join(project_root, "yolov8l.pt")
    data_yaml = os.path.join(project_root, "configs/datasets/gtsdb_yolov8.yaml")
    exp_dir = os.path.join(project_root, "experiments", "yolov8l_gtsdb")
    
    batch_size = 16
    success = False
    
    while batch_size >= 2 and not success:
        print(f"\n--- Attempting training with batch_size={batch_size} ---")
        try:
            model = YOLO(model_path)
            model.train(
                data=data_yaml,
                epochs=100,
                imgsz=416,
                batch=batch_size,
                device=0,
                project=exp_dir,
                name="run",
                exist_ok=True,
                seed=42
            )
            success = True
            print(f"Training succeeded with batch_size={batch_size}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM encountered with batch_size={batch_size}. Reducing batch size.")
                batch_size = batch_size // 2
                # Clean up memory
                if 'model' in locals():
                    del model
                gc.collect()
                torch.cuda.empty_cache()
            else:
                raise e
                
    if not success:
        print("Training failed even with minimal batch size.")
        return False
        
    return batch_size

def evaluate(batch_size):
    project_root = "/home/ms/Desktop/AntiStyler"
    exp_dir = os.path.join(project_root, "experiments", "yolov8l_gtsdb")
    weights = os.path.join(exp_dir, "run", "weights", "best.pt")
    data_yaml = os.path.join(project_root, "configs/datasets/gtsdb_yolov8.yaml")
    
    # 1. Run internal YOLO evaluation on test split
    print("\n--- Running internal YOLO test.py equivalent ---")
    model = YOLO(weights)
    metrics = model.val(
        data=data_yaml,
        split='test',
        imgsz=416,
        batch=batch_size,
        device=0,
        project=exp_dir,
        name="eval_clean",
        exist_ok=True,
        save_json=True
    )
    
    # 2. Convert and run COCO Evaluation
    import json
    yolov8_preds_json = os.path.join(exp_dir, "eval_clean", "predictions.json")
    gt_json = os.path.join(project_root, "evaluation_data", "gtsdb_coco", "instances_gtsdb_test.json")
    
    if os.path.exists(yolov8_preds_json):
        # Convert string IDs to int IDs
        with open(gt_json, 'r') as f:
            coco_gt = json.load(f)
            
        image_name_to_id = {}
        for img in coco_gt['images']:
            filename = img['file_name']
            stem = os.path.splitext(filename)[0]
            image_name_to_id[filename] = img['id']
            image_name_to_id[stem] = img['id']
            
        with open(yolov8_preds_json, 'r') as f:
            preds = json.load(f)
            
        fixed_preds = []
        for p in preds:
            img_str_id = p["image_id"]
            if img_str_id in image_name_to_id:
                p["image_id"] = image_name_to_id[img_str_id]
                fixed_preds.append(p)
                
        fixed_preds_path = os.path.join(exp_dir, "eval_clean", "predictions_fixed.json")
        with open(fixed_preds_path, 'w') as f:
            json.dump(fixed_preds, f)
            
        print("\n--- Running COCO Evaluation ---")
        import sys
        sys.path.append(project_root)
        from scripts.evaluate_coco_gtsdb import evaluate_coco
        stats = evaluate_coco(gt_json, fixed_preds_path)
        print(f"\nCOCO mAP@0.50: {stats[1]}")
    else:
        print("predictions.json not found!")

if __name__ == "__main__":
    final_batch = train_with_fallback()
    if final_batch:
        evaluate(final_batch)
