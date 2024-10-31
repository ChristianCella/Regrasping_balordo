from pathlib import Path
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import supervision as sv
import os
import matplotlib.pyplot as plt

from YOLO_simple_demo import YOLO_detect

# Load the model and the weights for segmentation
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_l"
IMAGE_PATH = Path("Images/Bottle1.png")
CHECKPOINT_PATH = Path("weights/sam_vit_l_0b3195.pth")

#######################DUMMY DETCTION#######################
results = YOLO_detect(IMAGE_PATH)
###########################################################

# Get the object detection results and save the bounding box coordinates
box = results[0].boxes.to('cpu')
box_coord = box.xyxyn.numpy() # shape: (n, 4) where n is the number of bounding boxes
print(box_coord)

# Get model
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

# Pass the model to the SamAutomaticMaskGenerator class 
mask_generator = SamAutomaticMaskGenerator(sam)

# Read the image and convert to rgb format
image = cv2.imread(str(IMAGE_PATH))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# sv.plot_image(image)

# Generate masks for the image using the mask generator
# sam_result = mask_generator.generate(image_rgb)
# print(sam_result[0].keys())

# Initialize sam predictor
predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

# Predict the mask for the bounding box
masks, scores, logits = predictor.predict(
    box = box_coord,
    multimask_output = False
)

# plot results
plt.imshow(image_rgb)
for mask in masks:
    plt.imshow(mask, alpha=0.5, cmap="jet")  # Overlay each mask with transparency
plt.axis("off")
plt.show()