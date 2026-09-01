# GTSDB - German Traffic Sign Detection Benchmark > augmented-Roboflow-ACCURATE-model
https://universe.roboflow.com/mohamed-traore-2ekkp/gtsdb---german-traffic-sign-detection-benchmark

Provided by a Roboflow user
License: CC BY 4.0

### This project was created by downloading the GTSDB German Traffic Sign Detection Benchmark
### dataset from Kaggle and importing the annotated training set files (images and annotation files)
### to Roboflow.
#### https://www.kaggle.com/datasets/safabouguezzi/german-traffic-sign-detection-benchmark-gtsdb
* Original home of the dataset: https://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset - [Institut Für Neuroinformatik](https://www.ini.rub.de/)

The annotation files were adjusted to conform to the [YOLO Keras TXT format](https://roboflow.com/formats/yolo-keras-txt) prior to upload, as the original format did not include a [label map file](https://blog.roboflow.com/label-map/).

`v1` contains the original imported images, without augmentations. This is the version to download and import to your own project if you'd like to add your own augmentations.

`v2` contains an augmented version of the dataset, with annotations. This version of the project was trained with Roboflow's "FAST" model.

`v3` contains an augmented version of the dataset, with annotations. This version of the project was trained with Roboflow's "ACCURATE" model.

* [Choosing Between Computer Vision Model Sizes](https://blog.roboflow.com/computer-vision-model-tradeoff/) | [New and Improved Roboflow Train](https://blog.roboflow.com/new-and-improved-roboflow-train/)