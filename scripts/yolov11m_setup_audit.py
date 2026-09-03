import os
import torch
import hashlib
from ultralytics import YOLO
import yaml

def get_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def main():
    print("=== YOLOv11-M SETUP AUDIT ===")
    
    # 1. Download and check weights
    model_name = "yolo11m.pt"
    print(f"Loading {model_name}...")
    model = YOLO(model_name)
    
    model_path = model.ckpt_path
    if not model_path:
        model_path = os.path.abspath(model_name)
    
    print(f"Checkpoint path: {model_path}")
    print(f"Checkpoint size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    
    ckpt_hash = get_hash(model_path)
    print(f"SHA256: {ckpt_hash}")
    
    # 2. Architecture and parameters
    info = model.info(verbose=False)
    # info returns (num_layers, num_params, num_gradients, flops) in ultralytics 8.3
    if isinstance(info, tuple) and len(info) >= 2:
        num_params = info[1]
    else:
        num_params = sum(x.numel() for x in model.model.parameters())
    print(f"Parameter count: {num_params}")
    
    print(f"Architecture: {model.model.__class__.__name__}")
    
    # 3. Hardware & Environment
    print(f"PyTorch version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA available: {cuda_avail}")
    if cuda_avail:
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU memory: {mem:.2f} GB")
        
    # 4. Inference Smoke Test (VRAM feasibility)
    print("Running inference smoke test...")
    try:
        model.to('cuda')
        dummy_input = torch.zeros((1, 3, 416, 416), device='cuda')
        with torch.no_grad():
            preds = model.model(dummy_input)
        print("CUDA inference: PASS")
        print("VRAM feasibility: PASS")
        print(f"Output shape (len/type): {type(preds)}")
        if isinstance(preds, (list, tuple)):
            print(f"preds[0] shape: {preds[0].shape}")
            if len(preds) > 1:
                print(f"preds[1] type: {type(preds[1])}")
        else:
            print(f"preds shape: {preds.shape}")
    except Exception as e:
        print("CUDA inference: FAIL")
        print(f"VRAM feasibility: FAIL ({e})")
        
    # 5. Dataset Audit
    dataset_yaml = "/home/ms/Desktop/AntiStyler/All Dataset/GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov11/data.yaml"
    print(f"Checking dataset: {dataset_yaml}")
    with open(dataset_yaml, 'r') as f:
        data_dict = yaml.safe_load(f)
        
    nc = data_dict.get('nc', -1)
    print(f"Number of classes (nc): {nc}")
    
    names = data_dict.get('names', [])
    class_14 = names[14] if len(names) > 14 else "NOT FOUND"
    print(f"Class 14 mapping: {class_14}")
    
    test_img_dir = os.path.join(os.path.dirname(dataset_yaml), "test/images")
    import glob
    test_imgs = glob.glob(os.path.join(test_img_dir, "*.jpg"))
    print(f"Test images found: {len(test_imgs)}")
    
    test_lbl_dir = os.path.join(os.path.dirname(dataset_yaml), "test/labels")
    test_lbls = glob.glob(os.path.join(test_lbl_dir, "*.txt"))
    total_annotations = 0
    for lbl in test_lbls:
        with open(lbl, 'r') as f:
            lines = f.readlines()
            total_annotations += len(lines)
    print(f"Test annotations found: {total_annotations}")

if __name__ == "__main__":
    main()
