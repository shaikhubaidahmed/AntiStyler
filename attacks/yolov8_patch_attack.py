import torch
import torch.nn as nn
import numpy as np

class YOLOv8PatchAttack:
    def __init__(self, model, config):
        """
        Adversarial Patch Attack for YOLOv8 (Anchor-free, DFL).
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
        self.target_class = config.get('target_class', 0)
        
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
        
        # Determine patch placement (center-ish / bottom-right-ish as in AntiStyler demo)
        _, _, H, W = image.shape
        min_y = int((H + self.patch_size) / 2)
        min_x = int((W + self.patch_size) / 2)
        
        # Clamp coordinates to ensure patch stays inside
        min_y = min(min_y, H - self.patch_size)
        min_x = min(min_x, W - self.patch_size)
        
        # YOLOv8 typically outputs strides [8, 16, 32]
        strides = [8, 16, 32]
        masks = []
        for s in strides:
            grid_h = int(np.ceil(H / s))
            grid_w = int(np.ceil(W / s))
            
            mask = torch.zeros((1, 1, grid_h, grid_w), device=device) # [1, 1, H, W]
            
            # Map patch physical coordinates to grid cell coordinates
            start_y = int(min_y / s)
            end_y = int(np.ceil((min_y + self.patch_size) / s))
            start_x = int(min_x / s)
            end_x = int(np.ceil((min_x + self.patch_size) / s))
            
            # Clamp to grid bounds
            start_y, end_y = max(0, start_y), min(grid_h, end_y)
            start_x, end_x = max(0, start_x), min(grid_w, end_x)
            
            mask[:, :, start_y:end_y, start_x:end_x] = 1.0
            masks.append(mask)

        # For YOLOv8, we set to train mode to get raw predictions [P3, P4, P5] without inference decoding
        # But we must freeze BatchNorm stats by setting BN modules to eval mode
        self.model.model.train()
        for m in self.model.model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
        for param in self.model.model.parameters():
            param.requires_grad = False
            
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            
            # Apply patch
            attacked_image = image.clone()
            attacked_image[:, :, min_y:min_y+self.patch_size, min_x:min_x+self.patch_size] = patch
            
            # Forward pass
            preds = self.model.model(attacked_image)
            
            if isinstance(preds, tuple) and len(preds) == 2:
                # It's in eval mode, extract the feature maps
                train_out = preds[1]
            else:
                train_out = preds
                
            if not isinstance(train_out, (list, tuple)):
                raise ValueError(f"Expected list/tuple, got {type(train_out)}")
                
            loss = 0
            for i, out_scale in enumerate(train_out):
                # out_scale shape: [1, 110, grid_h, grid_w]
                # Classes start at channel 64. Target class index is 64 + self.target_class
                cls_logits = out_scale[:, 64 + self.target_class : 64 + self.target_class + 1, :, :]
                
                # Apply sigmoid to get probability
                cls_prob = torch.sigmoid(cls_logits)
                
                # Apply spatial mask
                mask = masks[i]
                
                # Maximize cls probability at patch location (minimize negative)
                masked_score = cls_prob * mask
                
                if mask.sum() > 0:
                    loss -= masked_score.sum() / mask.sum()
                
            loss.backward(retain_graph=True)
            optimizer.step()
            
            # Project patch to [0, 1]
            patch.data = torch.clamp(patch.data, 0, 1)
            
        # Final attacked image
        attacked_image = image.clone()
        attacked_image[:, :, min_y:min_y+self.patch_size, min_x:min_x+self.patch_size] = patch.detach()
        
        return attacked_image.detach(), patch.detach()
