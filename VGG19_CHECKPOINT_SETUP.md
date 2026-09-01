# VGG19 Checkpoint Setup

The AntiStyler defense relies on a pretrained VGG-19 feature extractor. In an offline environment, this checkpoint must be side-loaded or provided locally to avoid download errors.

## Checkpoint Properties
- **Architecture:** VGG-19 (Standard ImageNet pretrained weights)
- **Format:** PyTorch state dictionary (`.pth`)
- **Checksum (SHA-256):** `dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44`
- **File Size:** `574,673,361` bytes

## Local Location
The valid local checkpoint has been located at:
`/home/ms/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth`

## Side-loading Instructions
If you need to replicate this on another offline machine:
1. On an online machine, run:
   ```python
   import torchvision.models as models
   models.vgg19(weights='VGG19_Weights.IMAGENET1K_V1')
   ```
   This will download the checkpoint into the PyTorch cache.
2. Copy the resulting `vgg19-dcbb9e9d.pth` file.
3. Transfer the file to the offline machine.
4. Place the file in the designated cache directory (e.g., `~/.cache/torch/hub/checkpoints/`) or specify the absolute path in `configs/antistyler.yaml` using the `vgg19_weights` key.
5. Verify the integrity using:
   ```bash
   sha256sum <path_to_checkpoint>
   ```
