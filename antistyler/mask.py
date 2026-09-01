import torch

class Mask:
    """
    Implements the masking of the original input image.
    The final mask is applied to the ORIGINAL INPUT IMAGE.
    
    The conceptual operation is:
    final_mask -> negative mask -> original input image -> defended image
    """
    def __init__(self, config: dict):
        pass

    def apply_mask(self, original_input: torch.Tensor, final_mask: torch.Tensor) -> torch.Tensor:
        """
        Applies the negative of the final mask to the original input image.
        Returns the defended image where the masked region is zeroed out (value 0).
        original_input: (B, C, H, W)
        final_mask: (B, 1, H, W) binary mask where 1 indicates adversarial patch.
        """
        # Ensure final mask matches input channels if needed, though broadcasting works
        # The equation: defended = original - (mask * original)
        # This zeroes out the original image where mask == 1
        # And leaves original image intact where mask == 0
        defended_image = original_input - (final_mask * original_input)
        
        return defended_image
