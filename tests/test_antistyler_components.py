import torch
import unittest
from antistyler.utils import add_random_padding, remove_padding
from antistyler.filter import Filter
from antistyler.enhancement import Enhancement
from antistyler.mask import Mask

class TestAntiStylerComponents(unittest.TestCase):

    def setUp(self):
        self.B, self.C, self.H, self.W = 2, 3, 32, 32
        self.padding_size = 10
        self.image = torch.rand((self.B, self.C, self.H, self.W))
        
        self.config = {
            "padding_size": 10,
            "raw_mask_percentile": 0.99,
            "dilation_kernel_1": 11,
            "erosion_kernel": 11,
            "mean_filter_kernel": 51,
            "binary_threshold": 0.5,
            "dilation_kernel_2": 11
        }

    def test_padding_dimensions(self):
        padded = add_random_padding(self.image, self.padding_size, "uniform")
        self.assertEqual(padded.shape, (self.B, self.C, self.H + 20, self.W + 20))
        
        removed = remove_padding(padded, self.padding_size)
        self.assertEqual(removed.shape, self.image.shape)
        
        # Test original image preservation inside padding
        self.assertTrue(torch.allclose(removed, self.image))

    def test_random_seed_deterministic(self):
        padded1 = add_random_padding(self.image, self.padding_size, "uniform", seed=42)
        padded2 = add_random_padding(self.image, self.padding_size, "uniform", seed=42)
        self.assertTrue(torch.allclose(padded1, padded2))
        
    def test_padding_range(self):
        padded = add_random_padding(self.image, self.padding_size, "uniform")
        # Ensure it is in [0, 1]
        self.assertTrue(torch.all(padded >= 0.0) and torch.all(padded <= 1.0))

    def test_filter_mask_dimensions_and_range(self):
        flt = Filter(self.config)
        padded = add_random_padding(self.image, self.padding_size, "uniform")
        anti_styled = padded.clone()
        # Introduce a fake patch
        anti_styled[:, :, 15:20, 15:20] = 1.0 
        
        raw_mask = flt.generate_raw_mask(padded, anti_styled)
        
        # 1. Mask dimensions should match ORIGINAL image dimensions (H, W)
        self.assertEqual(raw_mask.shape, (self.B, 1, self.H, self.W))
        
        # 2. Mask value range should be {0, 1}
        unique_vals = torch.unique(raw_mask)
        self.assertTrue(all(v.item() in [0.0, 1.0] for v in unique_vals))

    def test_enhancement_morphology(self):
        enh = Enhancement(self.config)
        raw_mask = torch.zeros((self.B, 1, self.H, self.W))
        raw_mask[:, :, 10:15, 10:15] = 1.0
        
        enhanced = enh.enhance(raw_mask)
        self.assertEqual(enhanced.shape, (self.B, 1, self.H, self.W))
        unique_vals = torch.unique(enhanced)
        self.assertTrue(all(v.item() in [0.0, 1.0] for v in unique_vals))
        
    def test_masking_original_preservation(self):
        mask_module = Mask(self.config)
        final_mask = torch.zeros((self.B, 1, self.H, self.W))
        # Mask a small 5x5 region
        final_mask[:, :, 5:10, 5:10] = 1.0
        
        defended = mask_module.apply_mask(self.image, final_mask)
        
        # 1. Final output dimensions
        self.assertEqual(defended.shape, self.image.shape)
        
        # 2. Original image preservation outside mask
        # We invert the mask to get regions outside the patch
        outside_mask = 1.0 - final_mask
        self.assertTrue(torch.allclose(defended * outside_mask, self.image * outside_mask))
        
        # 3. Defended image is zero inside the mask
        inside_mask = final_mask
        self.assertTrue(torch.allclose(defended * inside_mask, torch.zeros_like(defended)))

if __name__ == '__main__':
    unittest.main()
