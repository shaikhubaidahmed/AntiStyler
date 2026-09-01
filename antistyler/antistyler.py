import torch
from .config import load_config
from .utils import add_random_padding
from .style_removal import AntiStylerStyleRemoval
from .filter import Filter
from .enhancement import Enhancement
from .mask import Mask

class AntiStyler:
    """
    Standalone reproduction of the AntiStyler defense.
    Integrates Style Removal, Filter, Enhancement, and Masking phases.
    """
    def __init__(self, config_path: str = "configs/antistyler.yaml", device: torch.device = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.config = load_config(config_path)
        self.padding_size = self.config.get("padding_size", 10)
        self.style_distribution = self.config.get("style_distribution", "uniform")
        
        self.style_removal = AntiStylerStyleRemoval(self.config, self.device)
        self.filter = Filter(self.config)
        self.enhancement = Enhancement(self.config)
        self.mask_module = Mask(self.config)

    def defend(self, original_input: torch.Tensor, seed: int = None, debug: bool = False) -> torch.Tensor:
        """
        Defends the original input image against adversarial patch attacks.
        original_input: (B, C, H, W) float tensor [0,1] RGB
        Returns:
            defended_image: (B, C, H, W) float tensor [0,1] RGB
        """
        original_input = original_input.to(self.device)
        
        # Phase 1: Style Removal
        padded_input = add_random_padding(original_input, self.padding_size, self.style_distribution, seed)
        _, anti_styled_image = self.style_removal.remove_style(padded_input, seed)
        
        # Phase 2: Filter (Raw Mask)
        raw_mask = self.filter.generate_raw_mask(padded_input, anti_styled_image)
        
        # Phase 3: Enhancement
        final_mask = self.enhancement.enhance(raw_mask)
        
        # Phase 4: Mask (Defended Image)
        defended_image = self.mask_module.apply_mask(original_input, final_mask)
        
        if debug:
            return {
                "original_input": original_input.detach().cpu(),
                "padded_input": padded_input.detach().cpu(),
                "anti_styled_image": anti_styled_image.detach().cpu(),
                "raw_mask": raw_mask.detach().cpu(),
                "final_mask": final_mask.detach().cpu(),
                "defended_image": defended_image.detach().cpu()
            }
            
        return defended_image
