from PIL import Image, ImageChops
import random
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import os
import matplotlib
import numpy as np

# Data for LaTeX font
matplotlib.rcParams['mathtext.fontset'] = 'cm' # 'cm' or 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rc('xtick', labelsize = 15)
matplotlib.rc('ytick', labelsize = 15)

# Load object image (with black background)
object_img = Image.open(str(Path("Images_manipulation/No_background/output.jpg"))).convert("RGB")

# Create a mask to extract the object
mask = Image.eval(object_img.split()[2], lambda px: 0 if px < 100 else 255)
mask = mask.convert("RGB")

plt.figure(figsize = (10, 10))
plt.title("Background removed", fontsize = 20)
plt.imshow(mask.copy())
plt.axis('off')
plt.show()

# Load a random background image
background_files = glob(str(Path("Images_manipulation/Backgrounds/*.jpg")))
background_img = Image.open(random.choice(background_files)).resize(object_img.size)

# Mask in the correct format
new_mask = mask.convert("L")

# Composite the object onto the new background
output_img = Image.composite(object_img, background_img, new_mask)

# Display the image
plt.figure(figsize = (10, 10))
plt.title("Background removed", fontsize = 20)
plt.imshow(output_img)
plt.axis('off')
plt.show()

# Save or display the result
output_path = os.path.join(str(Path("Images_manipulation/Random_background")), "output.jpg")
bgr_img = cv2.cvtColor(np.array(output_img), cv2.COLOR_RGB2BGR)
cv2.imwrite(output_path, np.array(bgr_img))

