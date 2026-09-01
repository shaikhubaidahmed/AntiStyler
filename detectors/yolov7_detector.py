import sys
import os
import time
import torch
import numpy as np
import cv2

from detectors.base_detector import BaseDetector

# Add YOLOv7 source to path
YOLOV7_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "yolov7")
if YOLOV7_ROOT not in sys.path:
    sys.path.insert(0, YOLOV7_ROOT)

from models.experimental import attempt_load
from utils.general import non_max_suppression, check_img_size, scale_coords
from utils.torch_utils import select_device, time_synchronized
from utils.datasets import letterbox


class YOLOv7Detector(BaseDetector):
    """
    Wrapper around the official WongKinYiu/yolov7 implementation.
    Provides a clean API for model loading, prediction, and evaluation.
    
    Tensor conventions:
    - Input images: numpy HWC BGR uint8 [0, 255] (OpenCV convention)
      OR torch tensor BCHW RGB float [0, 1]
    - Internal processing: BCHW RGB float [0, 1]
    - Output bounding boxes: xyxy format in pixel coordinates
    """

    def __init__(self):
        self.model = None
        self.device = None
        self.stride = None
        self.names = None
        self.img_size = 640
        self.half = False

    def load_model(self, weights_path: str, device: torch.device = None, img_size: int = 640, half: bool = False):
        """
        Load YOLOv7 model weights.
        
        Args:
            weights_path: Path to yolov7.pt checkpoint
            device: torch.device, defaults to cuda if available
            img_size: inference image size (must be multiple of stride)
            half: use FP16 inference (only on CUDA)
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"YOLOv7 checkpoint not found: {weights_path}")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = attempt_load(weights_path, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.img_size = check_img_size(img_size, s=self.stride)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        
        self.half = half and self.device.type != 'cpu'
        if self.half:
            self.model.half()

        # Warmup
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters())))

        self.model.eval()
        print(f"YOLOv7 loaded: {weights_path}")
        print(f"  Classes: {len(self.names)}")
        print(f"  Stride: {self.stride}")
        print(f"  Image size: {self.img_size}")
        print(f"  Device: {self.device}")
        print(f"  Half: {self.half}")

    def _preprocess(self, image):
        """
        Preprocess a single image (numpy HWC BGR uint8 or RGB float tensor).
        Returns: (preprocessed_tensor, original_image, ratio_pad)
        """
        if isinstance(image, np.ndarray):
            # numpy HWC BGR uint8
            img0 = image.copy()
            img = letterbox(img0, self.img_size, stride=self.stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()
            img /= 255.0
        elif isinstance(image, torch.Tensor):
            # torch BCHW RGB float [0, 1] or CHW
            if image.dim() == 3:
                image = image.unsqueeze(0)
            img0 = (image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
            img = letterbox(img0, self.img_size, stride=self.stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()
            img /= 255.0
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        return img, img0

    def predict(self, image, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Run inference on a single image.
        
        Args:
            image: numpy HWC BGR uint8 or torch tensor
            conf_threshold: confidence threshold for NMS
            iou_threshold: IoU threshold for NMS
            
        Returns:
            detections: tensor of shape (N, 6) with [x1, y1, x2, y2, conf, cls]
                        in original image pixel coordinates
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        img, img0 = self._preprocess(image)

        with torch.no_grad():
            pred = self.model(img)[0]

        pred = non_max_suppression(pred, conf_threshold, iou_threshold)

        det = pred[0]
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            return det.cpu()
        else:
            return torch.zeros((0, 6))

    def get_predictions(self, image, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Get structured predictions.
        
        Returns:
            list of dicts with keys: bbox (x1,y1,x2,y2), confidence, class_id, class_name
        """
        det = self.predict(image, conf_threshold, iou_threshold)
        results = []
        for *xyxy, conf, cls in det:
            cls_id = int(cls.item())
            results.append({
                "bbox": [x.item() for x in xyxy],
                "confidence": conf.item(),
                "class_id": cls_id,
                "class_name": self.names[cls_id] if cls_id < len(self.names) else f"class_{cls_id}"
            })
        return results

    def evaluate(self, data_yaml: str):
        """
        Evaluate model on a dataset. 
        Delegates to the official YOLOv7 test.py script.
        """
        raise NotImplementedError("Full evaluation will be implemented when needed for training.")

    def get_inference_time(self, image, num_runs: int = 100):
        """
        Measure average inference time in milliseconds.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        img, _ = self._preprocess(image)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                self.model(img)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        times = []
        for _ in range(num_runs):
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                self.model(img)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.time() - t0) * 1000)

        return np.mean(times)
