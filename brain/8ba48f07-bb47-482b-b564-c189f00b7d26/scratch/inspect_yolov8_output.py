import torch
from ultralytics import YOLO

model = YOLO("/home/ms/Desktop/AntiStyler/experiments/yolov8l_gtsdb/run/weights/best.pt")
model.to('cuda')
dummy_img = torch.rand(1, 3, 416, 416, device='cuda')

model.model.train()
with torch.no_grad():
    preds = model.model(dummy_img)

print(f"Output type: {type(preds)}")
if isinstance(preds, list) or isinstance(preds, tuple):
    for i, p in enumerate(preds):
        if isinstance(p, torch.Tensor):
            print(f"Item {i} shape: {p.shape}")
        else:
            print(f"Item {i} type: {type(p)}")
