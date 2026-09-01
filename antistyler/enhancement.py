import torch
import torch.nn.functional as F

class Enhancement:
    """
    Implements the spatial filtering to enhance the raw mask and remove noise.
    Operations in order:
    1. Dilation (kernel=11)
    2. Erosion (kernel=11)
    3. Mean filter (kernel=51)
    4. Binary threshold (>= 0.5)
    5. Final Dilation (kernel=11)
    """
    def __init__(self, config: dict):
        self.dilation_kernel_1 = config.get("dilation_kernel_1", 11)
        self.erosion_kernel = config.get("erosion_kernel", 11)
        self.mean_filter_kernel = config.get("mean_filter_kernel", 51)
        self.binary_threshold = config.get("binary_threshold", 0.5)
        self.dilation_kernel_2 = config.get("dilation_kernel_2", 11)

    def _dilate(self, mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """
        Dilation implemented as a max filter.
        """
        pad = kernel_size // 2
        return F.max_pool2d(mask, kernel_size, stride=1, padding=pad)

    def _erode(self, mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """
        Erosion implemented as a min filter using max_pool2d on inverted mask.
        """
        pad = kernel_size // 2
        inverted_mask = 1.0 - mask
        eroded_inverted = F.max_pool2d(inverted_mask, kernel_size, stride=1, padding=pad)
        return 1.0 - eroded_inverted

    def _mean_filter(self, mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """
        Mean filter implemented using convolution with uniform weights.
        """
        pad = kernel_size // 2
        # mask is (B, 1, H, W)
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device) / (kernel_size * kernel_size)
        return F.conv2d(mask, kernel, padding=pad)

    def enhance(self, raw_mask: torch.Tensor) -> torch.Tensor:
        """
        Enhances the raw mask.
        raw_mask should be (B, 1, H, W) float tensor.
        """
        # 1. Dilation
        mask = self._dilate(raw_mask, self.dilation_kernel_1)
        
        # 2. Erosion
        mask = self._erode(mask, self.erosion_kernel)
        
        # 3. Mean Smoothing Filter
        mask = self._mean_filter(mask, self.mean_filter_kernel)
        
        # 4. Binary Threshold
        mask = (mask >= self.binary_threshold).float()
        
        # 5. Final Dilation
        final_mask = self._dilate(mask, self.dilation_kernel_2)
        
        return final_mask
