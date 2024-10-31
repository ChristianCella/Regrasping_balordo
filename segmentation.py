from pathlib import Path
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import supervision as sv
import os
import matplotlib.pyplot as plt
import numpy as np

from YOLO_simple_demo import YOLO_detect

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)  

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
box_coord = box.xyxy.numpy() # shape: (n, 4) where n is the number of bounding boxes

# Generate input point as center of bounding box
x = (box_coord[0][0] + box_coord[0][2]) / 2
y = (box_coord[0][1] + box_coord[0][3]) / 2
bbox_center = np.array([[x, y]])

input_point = bbox_center
input_label = np.array([1]) # 1 for foreground, 0 for background

print('-------------------DEBUG INFO------------------------')
print('Box coordinates: ', box_coord)
print('pixel coordinates for SAM model: ', bbox_center)
print('-----------------------------------------------------')

# Get model
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

# Pass the model to the SamAutomaticMaskGenerator class 
mask_generator = SamAutomaticMaskGenerator(sam)

# Read the image and convert to rgb format
image = cv2.imread(str(IMAGE_PATH))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize sam predictor
predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

# Predict the mask for the bounding box
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output = True
)

# plot results
for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()  