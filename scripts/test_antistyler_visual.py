import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys

# Ensure antistyler is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from antistyler.antistyler import AntiStyler
except RuntimeError as e:
    if "VGG19 PRETRAINED WEIGHTS NOT AVAILABLE LOCALLY" in str(e):
        print("VGG19 PRETRAINED WEIGHTS NOT AVAILABLE LOCALLY")
        print("Visual test cannot execute.")
        sys.exit(1)
    else:
        raise e

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Image {args.image} not found.")
        sys.exit(1)

    os.makedirs("debug_outputs", exist_ok=True)

    # Try initializing AntiStyler
    try:
        defense = AntiStyler(config_path="configs/antistyler.yaml")
    except RuntimeError as e:
        if "VGG19 PRETRAINED WEIGHTS NOT AVAILABLE LOCALLY" in str(e):
            print("VGG19 PRETRAINED WEIGHTS NOT AVAILABLE LOCALLY")
            print("Visual test cannot execute.")
            sys.exit(1)
        else:
            raise e

    # Load image
    img = Image.open(args.image).convert("RGB")
    transform = transforms.ToTensor() # Converts to [0, 1]
    input_tensor = transform(img).unsqueeze(0) # (1, C, H, W)

    print("Running AntiStyler in debug mode...")
    outputs = defense.defend(input_tensor, seed=42, debug=True)

    # Save outputs
    to_pil = transforms.ToPILImage()
    
    def save_tensor(t, name):
        # t is (1, C, H, W)
        t = t.squeeze(0)
        # If 1 channel, ToPILImage handles it as grayscale
        img = to_pil(t)
        img.save(os.path.join("debug_outputs", name))
        print(f"Saved {name}")

    save_tensor(outputs["original_input"], "1_original_input.png")
    save_tensor(outputs["padded_input"], "2_padded_input.png")
    save_tensor(outputs["anti_styled_image"], "4_anti_styled_image.png")
    save_tensor(outputs["raw_mask"], "6_raw_mask.png")
    save_tensor(outputs["final_mask"], "8_final_mask.png")
    save_tensor(outputs["defended_image"], "9_defended_image.png")

    print("Visual test complete.")

if __name__ == "__main__":
    main()
