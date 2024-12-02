# This script is used to enhance the images before the inference (Thank you Ale)

import fonts # TO display colored messags in the terminal
import cv2
from pathlib import Path
import os

# Define the path to the input image
input_rgb_path = str(Path("Images/Augmented_dataset_beckers/output_14.jpg"))

# Load the input image for processing
print(f'{fonts.red_light}Loading the image...{fonts.reset}')
image = cv2.imread(input_rgb_path)
print(f'{fonts.red_light}Enhancing the image...{fonts.reset}')

# Enhance the image using CLAHE and other techniques to improve contrast
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# Split the LAB image into separate channels (L, A, B)
l_channel, a_channel, b_channel = cv2.split(lab)

# Apply CLAHE to the L-channel to enhance contrast
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
#l_channel = clahe.apply(l_channel)

# (Optional) Apply histogram equalization to further stretch contrast
#l_channel = cv2.equalizeHist(l_channel)

# Merge the enhanced L-channel back with the original A and B channels
lab = cv2.merge((l_channel, a_channel, b_channel))

# Convert the LAB image back to the BGR color space
enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# (Optional) Further increase contrast using convertScaleAbs
alpha = 1.5  # Contrast control
beta = 2    # Brightness control
enhanced_image = cv2.convertScaleAbs(enhanced_image, alpha=alpha, beta=beta)

# (Optional) Apply Unsharp Masking for edge enhancement
blurred = cv2.GaussianBlur(enhanced_image, (0, 0), 3)
enhanced_image = cv2.addWeighted(enhanced_image, 1.5, blurred, -0.5, 0)

# Save the enhanced image to a new file
output_directory = str(Path("Images/Enhanced_images"))
input_filename = os.path.basename(input_rgb_path)

output_filename = f"enhanced_{input_filename}"
output_path = os.path.join(output_directory, output_filename)

cv2.imwrite(output_path, enhanced_image)
print(f'{fonts.red_light}The enhanced image has been saved as:', output_path, f'{fonts.reset}')
print(f'{fonts.yellow}oh wow, This is the object to segment! Close the image and I will predict! {fonts.reset}')
