import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

def add_random_padding(image: torch.Tensor, padding_size: int, distribution: str, seed: int = None) -> torch.Tensor:
    """
    Adds random-value padding to an image tensor.
    Image tensor should be shape (B, C, H, W) in range [0, 1].
    PAPER/NOTEBOOK fallback: RGB, float.
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    B, C, H, W = image.shape
    padded_H = H + 2 * padding_size
    padded_W = W + 2 * padding_size
    
    padded_image = torch.zeros((B, C, padded_H, padded_W), device=image.device, dtype=image.dtype)
    
    if distribution == "uniform":
        # Uniform [0, 1) as specified in PAPER
        padded_image[:, :, :padding_size, :] = torch.rand((B, C, padding_size, padded_W), device=image.device, dtype=image.dtype)
        padded_image[:, :, -padding_size:, :] = torch.rand((B, C, padding_size, padded_W), device=image.device, dtype=image.dtype)
        padded_image[:, :, :, :padding_size] = torch.rand((B, C, padded_H, padding_size), device=image.device, dtype=image.dtype)
        padded_image[:, :, :, -padding_size:] = torch.rand((B, C, padded_H, padding_size), device=image.device, dtype=image.dtype)
    else:
        # Fallback to normal (randn)
        padded_image[:, :, :padding_size, :] = torch.randn((B, C, padding_size, padded_W), device=image.device, dtype=image.dtype)
        padded_image[:, :, -padding_size:, :] = torch.randn((B, C, padding_size, padded_W), device=image.device, dtype=image.dtype)
        padded_image[:, :, :, :padding_size] = torch.randn((B, C, padded_H, padding_size), device=image.device, dtype=image.dtype)
        padded_image[:, :, :, -padding_size:] = torch.randn((B, C, padded_H, padding_size), device=image.device, dtype=image.dtype)
        
    padded_image[:, :, padding_size:-padding_size, padding_size:-padding_size] = image
    
    return padded_image

def remove_padding(image: torch.Tensor, padding_size: int) -> torch.Tensor:
    """
    Removes padding from an image tensor.
    """
    if padding_size > 0:
        return image[:, :, padding_size:-padding_size, padding_size:-padding_size]
    return image
