import torch
import sys
import os

project_root = "/home/ms/Desktop/AntiStyler"
best_ckpt = os.path.join(project_root, "experiments", "yolov11m_gtsdb", "run", "weights", "best.pt")

from ultralytics import YOLO

model = YOLO(best_ckpt)

# dummy input
dummy = torch.randn(1, 3, 416, 416).cuda()
model.model.cuda()
model.model.train()

# run forward
preds = model.model(dummy)

if isinstance(preds, tuple):
    print(f"Tuple of length {len(preds)}")
    if len(preds) == 2:
        train_out = preds[1]
    else:
        train_out = preds[0]
else:
    train_out = preds

if isinstance(train_out, list):
    for i, out in enumerate(train_out):
        print(f"Scale {i}: {out.shape}")
else:
    print(f"Out shape: {train_out.shape}")
