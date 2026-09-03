import torch
import torch.nn as nn
import numpy as np

class YOLOv7PatchAttackClass14Center:
    def __init__(self, model, config):
        """
        Adversarial Patch Attack for YOLOv7 (Class 14, Center coordinates).
        Config expects:
        - patch_size (int)
        - num_epochs (int)
        - lr (float)
        - target_class (int)
        """
        self.model = model
        self.patch_size = config.get('patch_size', 100)
        self.num_epochs = config.get('num_epochs', 200)
        self.lr = config.get('lr', 0.05)
        self.target_class = config.get('target_class', 14)
        
    def generate(self, image):
        """
        image: torch.Tensor (1, 3, H, W) in [0, 1]
        Returns:
        attacked_image: torch.Tensor
        patch: torch.Tensor
        """
        device = image.device
        
        # Initialize patch randomly
        patch = torch.nn.Parameter(
            torch.rand(1, 3, self.patch_size, self.patch_size, device=device),
            requires_grad=True
        )
        
        optimizer = torch.optim.Adam([patch], lr=self.lr)
        
        # FIXED CENTER COORDINATES
        min_x = 158
        min_y = 158
        
        # Create spatial masks for YOLOv7 heads dynamically based on model strides
        strides = self.model.stride.cpu().numpy() if hasattr(self.model, 'stride') else [8, 16, 32]
        masks = []
        for s in strides:
            _, _, H, W = image.shape
            grid_h = int(np.ceil(H / s))
            grid_w = int(np.ceil(W / s))
            
            mask = torch.zeros((1, 1, grid_h, grid_w, 1), device=device)
            # Map patch physical coordinates to grid cell coordinates
            start_y = int(min_y / s)
            end_y = int(np.ceil((min_y + self.patch_size) / s))
            start_x = int(min_x / s)
            end_x = int(np.ceil((min_x + self.patch_size) / s))
            
            # Clamp to grid bounds
            start_y, end_y = max(0, start_y), min(grid_h, end_y)
            start_x, end_x = max(0, start_x), min(grid_w, end_x)
            
            mask[:, :, start_y:end_y, start_x:end_x, :] = 1.0
            masks.append(mask)

        # For YOLOv7, we set to eval mode to get inference output and raw train_out
        self.model.eval()
        
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            
            # Apply patch
            attacked_image = image.clone()
            attacked_image[:, :, min_y:min_y+self.patch_size, min_x:min_x+self.patch_size] = patch
            
            # Forward pass
            preds = self.model(attacked_image, augment=False)
            
            if isinstance(preds, tuple) and len(preds) == 2:
                train_out = preds[1]
            else:
                train_out = preds
            
            loss = 0
            for i, out_scale in enumerate(train_out):
                # out_scale: [1, 3, h, w, 51]
                obj = torch.sigmoid(out_scale[..., 4:5]) # Keep last dim for broadcasting
                cls = torch.sigmoid(out_scale[..., 5 + self.target_class:6 + self.target_class])
                
                # Apply spatial mask for the current scale
                mask = masks[i]
                
                # Maximize obj * cls at the patch location (minimize negative)
                masked_score = obj * cls * mask
                # Compute mean only over the masked region to keep gradients properly scaled
                if mask.sum() > 0:
                    loss -= masked_score.sum() / mask.sum()
                
            loss.backward()
            optimizer.step()
            
            # Project patch to [0, 1]
            patch.data = torch.clamp(patch.data, 0, 1)
            
        # Final attacked image
        attacked_image = image.clone()
        attacked_image[:, :, min_y:min_y+self.patch_size, min_x:min_x+self.patch_size] = patch.detach()
        
        return attacked_image.detach(), patch.detach()
