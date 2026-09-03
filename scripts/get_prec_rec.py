import os
import shutil
import glob
from ultralytics import YOLO

project_root = "/home/ms/Desktop/AntiStyler"
base_out_dir = os.path.join(project_root, "experiments/yolov11m_antistyler_defense")
defended_dir = os.path.join(base_out_dir, "defended_images")

# Setup labels folder as sibling to images folder
labels_dir = os.path.join(base_out_dir, "labels")
os.makedirs(labels_dir, exist_ok=True)
test_label_dir = os.path.join(project_root, "All Dataset/GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov8/test/labels")
for label_file in glob.glob(os.path.join(test_label_dir, "*.txt")):
    shutil.copy(label_file, labels_dir)

# Also copy for clean sanity
clean_sanity_dir = os.path.join(base_out_dir, "clean_sanity_images")
# YOLO expects labels in the same relative path or replaced 'images' with 'labels'
# Since clean_sanity_dir is 'clean_sanity_images', labels should be 'clean_sanity_labels'
clean_sanity_labels_dir = os.path.join(base_out_dir, "clean_sanity_labels")
os.makedirs(clean_sanity_labels_dir, exist_ok=True)
for label_file in glob.glob(os.path.join(test_label_dir, "*.txt")):
    shutil.copy(label_file, clean_sanity_labels_dir)

model = YOLO(os.path.join(project_root, "experiments/yolov11m_gtsdb/run/weights/best.pt"))

print("Evaluating Defended...")
val_results_def = model.val(data=os.path.join(base_out_dir, "defended_data.yaml"), split='test', imgsz=416, batch=16, plots=False)
print(f"DEFENDED P: {val_results_def.box.mp} R: {val_results_def.box.mr}")

print("Evaluating Clean Sanity...")
val_results_clean = model.val(data=os.path.join(base_out_dir, "clean_sanity_data.yaml"), split='test', imgsz=416, batch=16, plots=False)
print(f"CLEAN SANITY P: {val_results_clean.box.mp} R: {val_results_clean.box.mr}")

