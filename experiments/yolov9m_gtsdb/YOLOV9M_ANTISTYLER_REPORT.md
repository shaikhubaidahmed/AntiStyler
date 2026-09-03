------------------------------------------------------------
YOLOv9-M ANTISTYLER DEFENSE
------------------------------------------------------------

Model:
YOLOv9-M

Checkpoint:
/home/ms/Desktop/AntiStyler/experiments/yolov9m_gtsdb/weights/best.pt

Checkpoint SHA256:
645721eea8b61c415f3b965ffa275d87de1ed0509bd844645656fb6ef124fb5c

Dataset:
GTSDB

Test images:
54

Ground-truth annotations:
82

Classes:
46

------------------------------------------------------------
MAIN RESULTS
------------------------------------------------------------

Metric                  Clean       Attacked       Defended

mAP@0.5                0.884       0.802          0.813
mAP@0.5:0.95           0.706       0.638          0.637
Precision              0.877       0.864          0.851
Recall                 0.797       0.699          0.725
AP75                   N/A         N/A            N/A
Detections              576         9175           1445
Inference ms/image      15.9        15.6           44.5
FPS                     62.89       64.1           6.03

------------------------------------------------------------
RECOVERY
------------------------------------------------------------

Attack mAP50 degradation:
0.082

Defense mAP50 recovery:
13.4146 %

Defended / Clean mAP50:
91.97 %

mAP50:95 recovery:
-1.4706 %

Precision recovery:
-100.0000 %

Recall recovery:
26.5306 %

------------------------------------------------------------
ANTISTYLER PROCESSING
------------------------------------------------------------

AntiStyler mean time/image:
121.3 ms

YOLOv9-M defended inference:
44.5 ms/image

Total defended pipeline:
165.7 ms/image

End-to-end FPS:
6.03

------------------------------------------------------------
CLEAN SANITY CHECK
------------------------------------------------------------

Clean mAP50:
0.884

Clean -> AntiStyler mAP50:
0.885

Change:
0.001

------------------------------------------------------------
14. VALIDATION AUDIT
------------------------------------------------------------

[PASS] Checkpoint SHA matches frozen checkpoint
[PASS] 54/54 attacked images processed
[PASS] 54/54 defended images generated
[PASS] One-to-one image correspondence
[PASS] 82 ground-truth annotations preserved
[PASS] 46-class mapping preserved
[PASS] No ground-truth modification
[PASS] No YOLOv9-M weight modification
[PASS] AntiStyler parameters unchanged
[PASS] Mask parameters unchanged
[PASS] Final mask applied to ORIGINAL attacked image
[PASS] COCO mAP50 evaluation valid
[PASS] mAP50:95 evaluation valid
[PASS] Precision valid
[PASS] Recall valid
[PASS] AP75 correctly reported as N/A if unavailable
[PASS] Detection count extracted from predictions
[PASS] Clean sanity check completed
[PASS] Reproducibility check completed
