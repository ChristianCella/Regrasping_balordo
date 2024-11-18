from PIL import Image, ImageEnhance
from pathlib import Path
import random

# Define a variable for the prints
verbose = True

# Directories
input_dir = Path("New_augmented_bottles")  # Folder with 217 composited images
output_dir = Path("New_enhanced_augmented")  # Folder to save enhanced images

# Ensure output directory exists
output_dir.mkdir(parents = True, exist_ok = True)

if verbose:
    print("Output directory:", output_dir)

# Load all images
input_files = list(input_dir.glob("*.jpg"))
if not input_files:
    raise FileNotFoundError("No images found in the 'Output_Images' folder!")

# Apply random enhancements to each image
for idx, input_file in enumerate(input_files):

    # Load the image
    img = Image.open(input_file).convert("RGB")
    
    # Randomly adjust brightness (0.8 to 1.2 range)
    brightness_enhancer = ImageEnhance.Brightness(img)
    img = brightness_enhancer.enhance(random.uniform(0.8, 1.2))
    
    # Randomly adjust saturation (0.8 to 1.2 range)
    saturation_enhancer = ImageEnhance.Color(img)
    img = saturation_enhancer.enhance(random.uniform(0.8, 1.2))
    
    # Save the enhanced image
    output_path = output_dir / f"enhanced_{idx}.jpg"
    img.save(output_path)

    print(f"Enhanced {idx + 1}/{len(input_files)}: Saved to {output_path}")
