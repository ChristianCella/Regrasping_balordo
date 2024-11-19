from PIL import Image
from rembg import remove
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import matplotlib
import os

# Data for LaTeX font
matplotlib.rcParams['mathtext.fontset'] = 'cm' # 'cm' or 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rc('xtick', labelsize = 15)
matplotlib.rc('ytick', labelsize = 15)

image_path = Path("Augmented_bottles/aug_0_54.png")

# Load the image
image = cv2.imread(str(image_path))
image_final = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
output_img = remove(image)
final_img = cv2.cvtColor(output_img.copy(), cv2.COLOR_BGR2RGB)

plt.figure(figsize = (10, 10))
plt.title("Background removed", fontsize = 20)
plt.imshow(final_img)
plt.axis('off')
plt.show()

# Save the image
output_path = os.path.join(str(Path("Images_manipulation/No_background")), "output.jpg")
cv2.imwrite(output_path, output_img)