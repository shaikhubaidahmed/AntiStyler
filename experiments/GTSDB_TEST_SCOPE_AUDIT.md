============================================================
GTSDB TEST-SET SCOPE AUDIT
============================================================

Dataset:
/home/ms/Desktop/AntiStyler/All Dataset/GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov8

Configured test split:
/home/ms/Desktop/AntiStyler/All Dataset/GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov8/test/images

Complete test images:
54

Complete test GT annotations:
82

Previous evaluation images:
54

Previous evaluation GT annotations:
82

54-image manifest:
/home/ms/Desktop/AntiStyler/experiments/GTSDB_54_IMAGE_MANIFEST.txt

54-image subset verified:
PASS

54-image selection mechanism:
The 54 images evaluated in all previous experiments natively comprise the COMPLETE `test` split provided by the Roboflow dataset export. No artificial restriction, random sampling, or code-based subsetting limit was enforced by the experimental pipeline. The YOLO validation scripts naturally process all 54 files located in the `test/images` directory.

Evidence:
Direct filesystem inspection confirms exactly 54 `.jpg` files and 54 `.txt` label files (containing 82 annotations) exist in the primary testing directory.

============================================================
SUBSET VS COMPLETE DATASET
============================================================

Metric                         54 subset       Complete test set

Images                        54             54
GT annotations                82             82
Classes represented           31             31
Mean annotations/image        1.52             1.52

Class coverage:
The evaluated set covers 100% of all ground-truth classes present in the designated test split (31 out of the 46 theoretical dataset classes are instantiated in the test split).

============================================================
CROSS-MODEL CONSISTENCY
============================================================

YOLOv7:
PASS

YOLOv8L:
PASS

YOLOv9M:
PASS

All three use identical 54-image identities:
PASS

============================================================
ROOT CAUSE
============================================================

Why were only 54 images evaluated?
Because the underlying dataset's test split folder physically contains exactly 54 images. 

Was this intentional?
YES (It naturally follows standard practice of evaluating the entire designated test set).

Was it scientifically justified by the current project protocol?
YES (Using the entire unmodified test split directly aligns with standard machine learning evaluation protocols).

============================================================
RESEARCH IMPACT
============================================================

Previous results are:
COMPLETE TEST RESULTS

Do previous models require rerun?
NO

Reason:
All previous experiments evaluated the full, intended, unmodified test split. There was no missing data, truncation, or accidental filtering.

============================================================
FUTURE PROTOCOL
============================================================

Recommended evaluation set:
COMPLETE TEST SET (which naturally equates to these 54 images)

Reason:
It is the official test split. Altering it now would break consistency with all prior baselines and experimental results. 

YOLOv11-M should proceed only after:
The dataset class mapping error (Class ID 14 -> "go right") identified in Prompt 19 is corrected.

============================================================
FINAL STATUS
============================================================

Dataset scope audit:
PASS

Evaluation protocol:
VALID

Critical blocker:
NONE
