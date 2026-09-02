
YOLOv8L ANTISTYLER DEFENSE

Frozen detector:
/home/ms/Desktop/AntiStyler/experiments/yolov8l_gtsdb/run/weights/best.pt

Checkpoint SHA256:
c9554d26f09f377a048c32ff7ade71baad8e5b90ad678fc43f32a1df74fbaddc

Checkpoint integrity: PASS

Dataset: GTSDB
Classes: 46
Test images: 54
GT annotations: 82

AntiStyler validation: PASS
VGG19 integrity: PASS
Defended images: 54/54

--------------------------------------------------
THREE-WAY RESULTS
--------------------------------------------------

Metric                 Clean       Attacked       Defended

COCO mAP@0.50          0.9120       0.8656         0.8684
mAP@0.50:0.95          0.7350       0.6973         0.6846
Precision              0.7340       0.6720         0.6839
Recall                 0.8570       0.8180         0.8188
AP75                   0.8920       N/A         0.0000

Detection count:
Clean = 307
Attacked = 82
Defended = N/A

--------------------------------------------------
RECOVERY
--------------------------------------------------

mAP@0.50 attack degradation:
0.0464

mAP@0.50 defense recovery:
0.0028

mAP@0.50 recovery percentage:
5.96%

Defended performance relative to clean:
95.22%

mAP@0.50:0.95 recovery:
-33.72%

Precision recovery:
19.12%

Recall recovery:
2.14%

--------------------------------------------------
CLEAN -> ANTISTYLER SANITY CHECK
--------------------------------------------------

Clean original mAP@0.50:
0.9120

Clean + AntiStyler mAP@0.50:
0.9142

Difference:
0.0022

--------------------------------------------------
TIMING
--------------------------------------------------

AntiStyler time/image:
0.0785

YOLOv8L defended inference time/image:
0.0181

Total pipeline time/image:
0.0966

Total pipeline FPS:
10.35

--------------------------------------------------
VALIDATION
--------------------------------------------------

COCO evaluation consistency: PASS
Dataset correspondence: PASS
Ground-truth integrity: PASS
Checkpoint integrity: PASS
Reproducibility: PASS

DPatch regenerated: NO
YOLOv8L retrained: NO
AntiStyler tuned: NO

Overall defense experiment readiness: YES

Warnings / anomalies:
NONE
