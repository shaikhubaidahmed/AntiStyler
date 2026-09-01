# GTSDB YOLOv7 Dataset Audit

## Dataset Source
- **Provider**: Roboflow (exported from Kaggle GTSDB)
- **Version**: v3-augmented-Roboflow-ACCURATE-model
- **Format**: YOLOv7 PyTorch (YOLO txt format)
- **License**: CC BY 4.0
- **Root Path**: `All Dataset/GTSDB - German Traffic Sign Detection Benchmark.v3-augmented-roboflow-accurate-model.yolov7pytorch/`

## Split Statistics

| Split | Images | Labels | Image-Label Match |
|-------|--------|--------|-------------------|
| train | 1149   | 1149   | ✅ Perfect         |
| valid | 108    | 108    | ✅ Perfect         |
| test  | 54     | 54     | ✅ Perfect         |
| **Total** | **1311** | **1311** | ✅ |

## Image Properties
- **Dimensions**: 416×416 (uniform across all splits)
- **Color Mode**: RGB
- **Format**: JPEG (.jpg)

## Annotation Properties
- **Format**: YOLO txt (`class_id x_center y_center width height`)
- **Total annotations**: 3,551
- **Classes**: 46 (IDs 0–45)
- **All 46 classes present**: ✅

## Annotation Validation
- **Corrupt annotations**: 0
- **Missing labels**: 0
- **Missing images**: 0
- **x_center out of [0,1]**: 0
- **y_center out of [0,1]**: 0
- **width out of (0,1]**: 0
- **height out of (0,1]**: 0
- **Invalid class IDs**: 0

## Class Names
```
0: ANIMALS, 1: CONSTRUCTION, 2: CYCLES CROSSING, 3: DANGER, 4: NO ENTRY,
5: PEDESTRIAN CROSSING, 6: SCHOOL CROSSING, 7: SNOW, 8: STOP,
9: bend left, 10: bend right, 11: bend, 12: give way,
13: go left or straight, 14: go left, 15: go right or straight,
16: go right, 17: go straight, 18: keep left, 19: keep right,
20: no overtaking (trucks), 21: no overtaking -trucks-, 22: no overtaking,
23: no traffic both ways, 24: no trucks, 25: priority at next intersection,
26: priority road, 27: restriction ends (overtaking (trucks)),
28: restriction ends (overtaking), 29: restriction ends -overtaking -trucks--,
30: restriction ends -overtaking-, 31: restriction ends 80,
32: restriction ends, 33: road narrows, 34: roundabout, 35: slippery road,
36: speed limit 100, 37: speed limit 120, 38: speed limit 20,
39: speed limit 30, 40: speed limit 50, 41: speed limit 60,
42: speed limit 70, 43: speed limit 80, 44: traffic signal, 45: uneven road
```

## Notes
- The dataset appears to be augmented by Roboflow (multiple variants of the same base image with different hashes).
- Duplicate class semantics detected: `no overtaking (trucks)` (ID 20) and `no overtaking -trucks-` (ID 21) appear to represent the same sign. Similarly for `restriction ends` variants (IDs 27-30). This is an artifact of the Roboflow export and should NOT be modified — it is the dataset we will use as-is for this research.

## Verdict
**DATASET VALID** — All annotations pass validation. No corrupted or missing files.
