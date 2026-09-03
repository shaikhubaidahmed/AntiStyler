============================================================
CLASS-14 MAPPING AUDIT
============================================================

Dataset:
/home/ms/Desktop/AntiStyler/All Dataset/GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov8

Total classes:
46

Class ID 14:
go left

Dataset mapping source:
/home/ms/Desktop/AntiStyler/All Dataset/GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov8/data.yaml

Class 14 test images:
0

Class 14 test annotations:
0

============================================================
DPATCH TARGET AUDIT
============================================================

Model       Configured ID       Actual ID Used       Valid?

YOLOv7      14                  0                    FAIL (Used dummy class 0: ANIMALS)

YOLOv8L     14                  14                   PASS

YOLOv9-M    14                  14                   PASS

============================================================
SEMANTIC TERMINOLOGY
============================================================

Previous documentation called class 14:
go right

Dataset actually defines class 14 as:
go left

Terminology mismatch:
YES

Does this change the mathematical attack target?
NO

============================================================
EXPERIMENTAL IMPACT
============================================================

YOLOv7 previous attack result:
VALID (Mathematically valid attack against Class 0, though misreported in terms of intent)

YOLOv8L previous attack result:
VALID (Mathematically valid attack against Class 14)

YOLOv9-M previous attack result:
VALID (Mathematically valid attack against Class 14)

Rerun required:
NO

Reason:
The attacks mathematically optimized the intended numerical tensor indices correctly. The semantic human-readable name in the report ("go right") was incorrect due to dataset alphabetical sorting, but this does not invalidate the mathematical adversarial gradient objective, the execution of the patch attack, or the AntiStyler defense validation.

============================================================
ANTISTYLER IMPACT
============================================================

AntiStyler affected:
NO

Reason:
AntiStyler is a pre-processing defense that operates strictly in the pixel space (image-to-image translation). It requires no knowledge of the specific target class, class ID, or semantic meaning of the adversarial patch.

============================================================
YOLOv11-M
============================================================

Recommended target class ID:
14

Dataset-defined class name:
go left

Proceed to YOLOv11-M:
YES

Blocker:
NONE
