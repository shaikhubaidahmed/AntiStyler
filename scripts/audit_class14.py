import os
import yaml
import glob
import re

def main():
    base_dir = "/home/ms/Desktop/AntiStyler"
    dataset_dir = os.path.join(base_dir, "All Dataset/GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov8")
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    
    names = data.get("names", [])
    print("=== FULL MAPPING ===")
    for i, n in enumerate(names):
        print(f"ID {i} -> {n}")
        
    print("\\n=== NEIGHBORING IDS ===")
    for i in range(12, 17):
        if i < len(names):
            print(f"ID {i} -> {names[i]}")
            
    print("\\n=== CLASS 14 STATS ===")
    target_class = 14
    splits = ["train", "valid", "test"]
    stats = {}
    
    for split in splits:
        lbl_dir = os.path.join(dataset_dir, f"{split}/labels" if split != "valid" else "valid/labels")
        lbls = glob.glob(os.path.join(lbl_dir, "*.txt"))
        count = 0
        img_count = 0
        images_with_14 = []
        for lbl in lbls:
            with open(lbl, "r") as f:
                lines = f.readlines()
                c14_in_file = 0
                for line in lines:
                    parts = line.strip().split()
                    if parts and int(parts[0]) == target_class:
                        c14_in_file += 1
                if c14_in_file > 0:
                    count += c14_in_file
                    img_count += 1
                    if split == "test":
                        images_with_14.append(os.path.basename(lbl).replace(".txt", ".jpg"))
        stats[split] = {"ann": count, "img": img_count, "files": images_with_14 if split == "test" else []}
        print(f"{split.upper()} -> {count} annotations across {img_count} images.")
        
    print(f"\\nTEST IMAGES CONTAINING CLASS 14:")
    for img in stats["test"]["files"]:
        print(img)
        
    print("\\n=== DPATCH CODE AUDIT ===")
    # v8 attack
    v8_attack_path = os.path.join(base_dir, "attacks/yolov8_patch_attack.py")
    if os.path.exists(v8_attack_path):
        with open(v8_attack_path, "r") as f:
            content = f.read()
            # find where target_class is used
            matches = re.findall(r"target_class", content)
            print(f"YOLOv8 patch attack uses 'target_class': {len(matches)} times.")
    
    # v9 attack
    v9_attack_path = os.path.join(base_dir, "attacks/yolov9_patch_attack.py")
    if os.path.exists(v9_attack_path):
        with open(v9_attack_path, "r") as f:
            content = f.read()
            matches = re.findall(r"target_class", content)
            print(f"YOLOv9 patch attack uses 'target_class': {len(matches)} times.")
            
    # Check runner scripts to see what class they passed
    runner_scripts = [
        "scripts/yolov7_attack_full_experiment.py",
        "scripts/yolov8l_attack_full_experiment.py",
        "scripts/yolov9_dpatch_experiment.py"
    ]
    for script in runner_scripts:
        script_path = os.path.join(base_dir, script)
        if os.path.exists(script_path):
            with open(script_path, "r") as f:
                content = f.read()
                matches = re.findall(r"target_class.*=.*", content)
                print(f"{script} -> {matches}")

if __name__ == "__main__":
    main()
