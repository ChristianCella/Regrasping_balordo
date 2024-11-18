from PIL import Image, ImageChops
from rembg import remove
import random
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from collections import Counter

# Define a variable for the prints
verbose = True

# Directories
input_images_dir = Path("Augmented_bottles")  # Folder with 217 images
output_dir = Path("New_augmented_bottles")  # Folder to save final images
backgrounds_dir = Path("Images_manipulation/Backgrounds")  # Folder with 10 backgrounds
temp_dir = Path("Images_manipulation/Temp")  # Temporary folder for background-removed images

# Ensure output directories exist
output_dir.mkdir(parents = True, exist_ok = True)
temp_dir.mkdir(parents = True, exist_ok = True)

if verbose:
    print("Output directory:", output_dir)
    print("Temporary directory:", temp_dir)

# Load background images
background_files = glob(str(backgrounds_dir / "*.jpg"))
if not background_files:
    raise FileNotFoundError("No background images found in the 'Backgrounds' folder!")

# Load input images
input_files = glob(str(input_images_dir / "*.png"))
if not input_files:
    raise FileNotFoundError("No input images found in the 'Input_Images' folder!")

# Ensure balanced use of backgrounds
background_counter = Counter()
num_backgrounds = len(background_files)

if verbose:
    print(f"Found {len(input_files)} input images.")
    print(f"Found {num_backgrounds} background images.")

# Process each input image
for idx, input_file in enumerate(input_files):

    # Load input image
    input_image = cv2.imread(input_file)
    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Remove background
    background_removed = remove(input_image)
    background_removed_rgb = cv2.cvtColor(background_removed, cv2.COLOR_BGR2RGB)
    
    # Save background-removed image temporarily
    temp_path = temp_dir / f"temp_{idx}.png"
    cv2.imwrite(str(temp_path), background_removed)

    # Load the processed image into PIL for further manipulation
    object_img = Image.open(temp_path).convert("RGB")

    # Create a binary mask
    mask = Image.eval(object_img.split()[2], lambda px: 0 if px < 90 else 255).convert("L")

    # Select a random background, ensuring balance
    selected_background = min(background_files, key=lambda bg: background_counter[bg])
    background_counter[selected_background] += 1

    # Load the selected background
    background_img = Image.open(selected_background).resize(object_img.size)

    # Composite the object onto the background
    output_img = Image.composite(object_img, background_img, mask)

    # Save the final composited image
    output_path = output_dir / f"output_{idx}.jpg"
    output_img.save(output_path)

    print(f"Processed {idx + 1}/{len(input_files)}: Saved to {output_path}")

# Verify background distribution
print("Background usage count:", background_counter)
