# AntiStyler Implementation Documentation

This document describes our standalone, research-grade reproduction of the AntiStyler defense based on the CVPR 2026 paper, the official notebook implementation, and our derived experimental contract.

## Architecture
The AntiStyler defense is a four-stage pipeline:
1. **Style Removal**: Removes random noise style associated with adversarial attacks.
2. **Filter**: Generates a raw binary mask by isolating the top changes introduced by Style Removal.
3. **Enhancement**: Applies sequential spatial morphology filters to refine the raw mask.
4. **Mask**: Applies the final mask to the ORIGINAL INPUT IMAGE to produce the defended output.
*Source: CONTRACT / PAPER (Section 4.1)*

## Style Removal
- **Content Image**: The padded input image.
- **Style Image**: A randomly generated noise image.
- **Output**: The anti-styled image with the random style minimized.
*Source: PAPER (Section 4.2)*

## VGG19 Layers
- **Feature Extractor**: Pre-trained VGG-19 (`vgg19(weights='VGG19_Weights.IMAGENET1K_V1')`).
- **Style Layers**: The first five convolutional layers (`conv_1` through `conv_5`).
- **Content Layer**: The fourth convolutional layer (`conv_4`).
*Source: PAPER (Section 5.1) / NOTEBOOK*

## Loss Functions
- **Style Loss**: Mean Squared Error (MSE) between the Gram matrices of the style layers.
- **Content Loss**: Mean Squared Error (MSE) between the feature maps of the content layer.
- **Total Loss**: $\alpha L_C - \beta L_S$ where $\alpha = 1.0$ and $\beta = 1000.0$.
*Source: PAPER (Section 4.2, 5.1) / NOTEBOOK*

## Optimization
- **Optimizer**: Adam
- **Learning Rate**: `0.05`
- **Steps**: 1
*Source: CONTRACT (Notebook Fallback for optimizer, Paper for steps)*

## Padding
- **Size**: 10 pixels on all sides.
- **Value**: Random noise sampled uniformly from `[0, 1)`.
- **Application**: The original input image is placed in the center of the padded tensor without resizing. The padding is removed before passing the raw mask to the enhancement stage.
*Source: PAPER (Section 5.1) / CONTRACT*

## Random Style
- **Distribution**: Uniform distribution over `[0, 1)`.
*Source: PAPER (Section 4.2)*

## Filter
- **Operation**: Absolute difference between the padded input and anti-styled image, followed by a channel mean.
- **Threshold**: Top 1% percentile difference (`tau = 0.99`). The mask is binary (1 for values $\ge$ threshold, 0 otherwise).
*Source: NOTEBOOK / PAPER (Section 4.3)*

## Enhancement
Sequential spatial filters applied to the raw mask:
1. **Dilation**: Max pool, `kernel=11`, `padding=5`.
2. **Erosion**: Min pool (implemented as `1 - max_pool2d(1 - mask)`), `kernel=11`, `padding=5`.
3. **Mean Filter**: Convolution with uniform weights, `kernel=51`, `padding=25`.
4. **Binary Threshold**: Values $\ge 0.5$ set to 1.
5. **Final Dilation**: Max pool, `kernel=11`, `padding=5`.
*Source: NOTEBOOK / CONTRACT*

## Mask
- **Target**: The final mask is applied to the ORIGINAL INPUT IMAGE.
- **Operation**: `defended_image = original_input - (final_mask * original_input)`.
*Source: PAPER (Section 4.4, Figure 1) / CONTRACT*

## Data Types
- **Images**: Float tensors in the range `[0, 1]`.
- **Masks**: Binary float tensors containing only values `0.0` or `1.0`.
*Source: NOTEBOOK FALLBACK / CONTRACT*

## Tensor Shapes
- All image operations maintain `(B, C, H, W)` shape conventions.
- Masks are `(B, 1, H, W)`.
*Source: NOTEBOOK FALLBACK*

## Reproducibility
- All stochastic operations (random padding, style image generation) accept an explicit `seed` parameter to ensure deterministic execution for testing.
*Source: CONTRACT*

## Known Limitations
- VGG-19 weights must be available locally. Automatic download is disabled in the offline environment.
- The pipeline assumes RGB float tensors in `[0,1]` without ImageNet normalization, reflecting the notebook's behavior.

## Paper-vs-Implementation Decisions
1. **Mask Target**: We explicitly mask the original input image as described in the paper, rather than the anti-styled image as coded in the notebook.
2. **Style Distribution**: We use a uniform distribution `[0, 1)` for the style image as stated in the paper, overriding the notebook's use of a normal distribution (`randn`).
3. **Optimization Steps**: We strictly use 1 optimization step, matching the paper and notebook cell 925.
