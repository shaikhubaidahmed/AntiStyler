import os
import glob
import json

def main():
    print("=== DATASET TEST SCOPE AUDIT ===")
    base_dir = "/home/ms/Desktop/AntiStyler"
    dataset_path = os.path.join(base_dir, "All Dataset/GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov8")
    
    test_img_dir = os.path.join(dataset_path, "test/images")
    test_lbl_dir = os.path.join(dataset_path, "test/labels")
    
    # 1. Complete dataset count
    test_imgs = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg")))
    test_lbls = sorted(glob.glob(os.path.join(test_lbl_dir, "*.txt")))
    
    print(f"Dataset path: {dataset_path}")
    print(f"Complete test images: {len(test_imgs)}")
    print(f"Complete test label files: {len(test_lbls)}")
    
    total_annotations = 0
    annotations_per_img = []
    classes_found = set()
    for lbl in test_lbls:
        with open(lbl, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            total_annotations += len(lines)
            annotations_per_img.append(len(lines))
            for line in lines:
                parts = line.split()
                if parts:
                    classes_found.add(int(parts[0]))
    
    print(f"Complete test annotations: {total_annotations}")
    if annotations_per_img:
        print(f"Min ann/img: {min(annotations_per_img)}, Max: {max(annotations_per_img)}, Mean: {sum(annotations_per_img)/len(annotations_per_img):.2f}")
    
    # 2. Identify the 54 images from previous experiments
    v9_attacked = sorted(glob.glob(os.path.join(base_dir, "experiments/yolov9m_gtsdb/attacked/images/*.jpg")))
    print(f"v9 attacked images count: {len(v9_attacked)}")
    
    subset_filenames = [os.path.basename(p) for p in v9_attacked]
    
    # Dump subset manifest
    manifest_path = os.path.join(base_dir, "experiments/GTSDB_54_IMAGE_MANIFEST.txt")
    with open(manifest_path, "w") as f:
        for fname in subset_filenames:
            f.write(fname + "\\n")
    print(f"Manifest saved to: {manifest_path}")
    
    # Check annotations for those 54 images
    subset_annotations = 0
    subset_classes = set()
    for fname in subset_filenames:
        lbl_fname = fname.replace(".jpg", ".txt")
        lbl_path = os.path.join(test_lbl_dir, lbl_fname)
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                subset_annotations += len(lines)
                for line in lines:
                    parts = line.split()
                    if parts:
                        subset_classes.add(int(parts[0]))
    
    print(f"Subset test annotations: {subset_annotations}")
    print(f"Subset classes represented: {len(subset_classes)}")
    
    # 3. Check v7 and v8
    v8_attacked = sorted(glob.glob(os.path.join(base_dir, "experiments/yolov8l_gtsdb/dpatch_full/images/*.jpg")))
    v8_fnames = [os.path.basename(p) for p in v8_attacked]
    print(f"v8 attacked images count: {len(v8_attacked)}")
    
    v7_attacked = sorted(glob.glob(os.path.join(base_dir, "experiments/yolov7_gtsdb/attacked/images/*.jpg")))
    v7_fnames = [os.path.basename(p) for p in v7_attacked]
    print(f"v7 attacked images count: {len(v7_attacked)}")
    
    print(f"v9 == v8 subset: {set(subset_filenames) == set(v8_fnames)}")
    print(f"v9 == v7 subset: {set(subset_filenames) == set(v7_fnames)}")

    # 4. Search for the root cause
    # I will do a quick grep over scripts to see why they selected 54.
    import subprocess
    print("Checking scripts for subset logic...")
    try:
        res = subprocess.run(["grep", "-rn", "54", "scripts/"], cwd=base_dir, capture_output=True, text=True)
        print("Grep for 54:\\n", res.stdout)
    except Exception as e:
        print("Grep failed", e)

if __name__ == "__main__":
    main()
