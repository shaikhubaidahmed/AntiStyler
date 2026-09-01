from abc import ABC, abstractmethod
import torch

class BaseDetector(ABC):
    """
    Abstract base class for object detectors used in this research project.
    All detector implementations must inherit from this class.
    """
    
    @abstractmethod
    def load_model(self, weights_path: str, device: torch.device = None):
        """Load model weights from a checkpoint file."""
        pass
    
    @abstractmethod
    def predict(self, image: torch.Tensor, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Run inference on an image tensor.
        Returns predictions as a list of [x1, y1, x2, y2, confidence, class_id].
        """
        pass
    
    @abstractmethod
    def evaluate(self, data_yaml: str):
        """
        Evaluate the model on a dataset defined by a YAML config.
        Returns evaluation metrics (e.g., mAP@0.5).
        """
        pass
    
    @abstractmethod
    def get_predictions(self, image, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Get structured predictions from an image.
        Returns a list of dicts with keys: bbox, confidence, class_id, class_name.
        """
        pass
    
    @abstractmethod
    def get_inference_time(self, image, num_runs: int = 100):
        """
        Measure average inference time over num_runs.
        Returns average time in milliseconds.
        """
        pass
