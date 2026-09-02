import torch
from ultralytics import YOLO

model = YOLO("/home/ms/Desktop/AntiStyler/experiments/yolov8l_gtsdb/run/weights/best.pt")
model.to('cuda')
dummy_img = torch.rand(1, 3, 416, 416, device='cuda')

model.model.eval()
with torch.no_grad():
    preds = model.model(dummy_img)

print(f"Output type: {type(preds)}")
if isinstance(preds, tuple):
    print(f"Length of tuple: {len(preds)}")
    for i, p in enumerate(preds):
        if isinstance(p, torch.Tensor):
            print(f"Item {i} shape: {p.shape}")
        elif isinstance(p, list) or isinstance(p, tuple):
            print(f"Item {i} is list/tuple of length {len(p)}")
            for j, p2 in enumerate(p):
                print(f"  SubItem {j} shape: {p2.shape}")
