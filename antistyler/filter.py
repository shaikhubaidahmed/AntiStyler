import torch
from .utils import remove_padding

class Filter:
    """
    Implements the raw mask generation according to the contract.
    padded_input -> absolute difference -> channel aggregation -> top 1% -> raw mask
    """
    def __init__(self, config: dict):
        self.tau = config.get("raw_mask_percentile", 0.99)
        self.padding_size = config.get("padding_size", 10)
        
    def generate_raw_mask(self, padded_input: torch.Tensor, anti_styled_image: torch.Tensor) -> torch.Tensor:
        """
        Generates raw mask from padded images.
        Returns:
            raw_mask: (B, 1, H, W) binary mask at original unpadded dimensions.
        """
        # 1. Absolute difference
        difference_image = torch.abs(anti_styled_image - padded_input)
        
        # 2. Channel aggregation (mean across RGB)
        difference_image = difference_image.mean(dim=1, keepdim=True) # (B, 1, padded_H, padded_W)
        
        # We need to process per batch item for thresholding
        raw_masks = []
        for i in range(difference_image.size(0)):
            diff = difference_image[i]
            
            # 3. Percentile thresholding
            flattened_values = diff.flatten()
            k = int(self.tau * flattened_values.shape[0])
            
            if k == 0:
                # Edge case if tensor is extremely small
                k = 1
            elif k > flattened_values.shape[0]:
                k = flattened_values.shape[0]
                
            threshold_value = torch.kthvalue(flattened_values, k, dim=0).values.item()
            
            # Binary threshold
            raw_mask = (diff >= threshold_value).float()
            raw_masks.append(raw_mask)
            
        raw_mask_batch = torch.stack(raw_masks, dim=0)
        
        # 4. Remove padding contribution before returning the mask
        raw_mask_cropped = remove_padding(raw_mask_batch, self.padding_size)
        
        return raw_mask_cropped
