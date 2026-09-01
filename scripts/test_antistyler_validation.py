import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from antistyler.antistyler import AntiStyler
from antistyler.config import load_config
import numpy as np

def validate_tensor(tensor, name, expected_shape):
    assert not torch.isnan(tensor).any(), f"{name} contains NaN values"
    assert not torch.isinf(tensor).any(), f"{name} contains Inf values"
    assert tensor.shape == expected_shape, f"{name} shape {tensor.shape} does not match expected {expected_shape}"

def main():
    print("Initializing AntiStyler...")
    defense = AntiStyler(config_path="configs/antistyler.yaml")
    
    B, C, H, W = 1, 3, 100, 100
    input_tensor = torch.rand((B, C, H, W))
    
    print("Running end-to-end inference (Seed 42)...")
    outputs_1 = defense.defend(input_tensor, seed=42, debug=True)
    
    padding_size = defense.padding_size
    padded_H, padded_W = H + 2 * padding_size, W + 2 * padding_size
    
    print("Validating outputs...")
    # 1. Input dimensions are preserved (handled by output checks)
    # 2. Output dimensions equal input dimensions
    validate_tensor(outputs_1["defended_image"], "Defended Image", (B, C, H, W))
    
    # 3. AntiStyled image is valid
    validate_tensor(outputs_1["anti_styled_image"], "AntiStyled Image", (B, C, padded_H, padded_W))
    
    # 4. Raw mask is valid
    validate_tensor(outputs_1["raw_mask"], "Raw Mask", (B, 1, H, W))
    
    # 5. Enhanced mask is not returned separately in debug, but we have final_mask
    
    # 6. Final mask is valid
    validate_tensor(outputs_1["final_mask"], "Final Mask", (B, 1, H, W))
    
    # 7. Defended image is valid (checked above)
    
    # 8. Pixel ranges are valid
    assert (outputs_1["defended_image"] >= 0).all() and (outputs_1["defended_image"] <= 1).all(), "Defended image out of range [0,1]"
    assert (outputs_1["anti_styled_image"] >= 0).all() and (outputs_1["anti_styled_image"] <= 1).all(), "AntiStyled image out of range [0,1]"
    assert (outputs_1["final_mask"] == 0).logical_or(outputs_1["final_mask"] == 1).all(), "Final mask is not binary"
    
    # 9, 10, 11 checked by validate_tensor
    
    # 13. Padding is correctly removed (mask is H, W not padded_H, padded_W)
    assert outputs_1["final_mask"].shape == (B, 1, H, W)
    
    # 14. Masking Rule Verification: pixels outside mask are identical to original input
    mask = outputs_1["final_mask"]
    outside_mask = 1.0 - mask
    defended = outputs_1["defended_image"]
    original = outputs_1["original_input"]
    
    # Tolerance for float point math
    diff = torch.abs((defended * outside_mask) - (original * outside_mask))
    assert torch.max(diff) < 1e-6, "Pixels outside mask do not perfectly match original input"
    print(f"Mask verification passed with max diff: {torch.max(diff)}")
    
    print("Running reproducibility test (Seed 42)...")
    outputs_2 = defense.defend(input_tensor, seed=42, debug=True)
    
    diff_reproducibility = torch.max(torch.abs(outputs_1["anti_styled_image"] - outputs_2["anti_styled_image"]))
    assert diff_reproducibility < 1e-5, f"Reproducibility failed. Max diff: {diff_reproducibility}"
    print(f"Reproducibility passed (Seed 42 == Seed 42) with max diff: {diff_reproducibility}")
    
    print("Running stochasticity test (Seed 99)...")
    outputs_3 = defense.defend(input_tensor, seed=99, debug=True)
    diff_stochasticity = torch.max(torch.abs(outputs_1["anti_styled_image"] - outputs_3["anti_styled_image"]))
    assert diff_stochasticity > 1e-5, "Stochasticity failed. Seed 99 produced identical output to Seed 42."
    print("Stochasticity passed (Seed 99 != Seed 42)")
    
    print("ALL TESTS PASSED")

if __name__ == "__main__":
    main()
