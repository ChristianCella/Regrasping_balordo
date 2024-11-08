from pathlib import Path
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from YOLO_simple_demo import YOLO_detect

from segment_anything_2.sam2.build_sam import build_sam2
from segment_anything_2.sam2.sam2_image_predictor import SAM2ImagePredictor

########################## AUXILIARY FUNCTIONS ############################
np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

#############################################################################

# Load the model and the weights for segmentation
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
IMAGE_PATH = "Images/Bottle1.png"
CHECKPOINT_PATH = "segment_anything_2/checkpoints/sam2.1_hiera_large.pt"

# check if paths exists
if not os.path.exists(CHECKPOINT_PATH):
    raise ValueError(f"Checkpoint path {CHECKPOINT_PATH} does not exist")

if not os.path.exists(IMAGE_PATH):
    raise ValueError(f"Image path {IMAGE_PATH} does not exist")

#######################DUMMY DETCTION#######################
results = YOLO_detect(IMAGE_PATH)
############################################################

# Get the object detection results and save the bounding box coordinates
box = results[0].boxes.to('cpu')
box_coord = box.xyxy.numpy() # shape: (n, 4) where n is the number of bounding boxes

# Generate input point as center of bounding box
x = (box_coord[0][0] + box_coord[0][2]) / 2
y = (box_coord[0][1] + box_coord[0][3]) / 2
bbox_center = np.array([[x, y]])

input_point = bbox_center
input_label = np.array([1]) # Points are input to the model in (x,y) format and come with labels 1 (foreground point) or 0 (background point)

print('-------------------DEBUG INFO------------------------')
print('Box coordinates: ', box_coord)
print('pixel coordinates for SAM model: ', bbox_center)
print('-----------------------------------------------------')

# Read the image and convert to rgb format
image = cv2.imread(str(IMAGE_PATH))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Plot image for debugging 
# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# plt.axis('on')
# plt.show()

# Load model and predictor
sam2_model = build_sam2(MODEL_CFG, CHECKPOINT_PATH, device=DEVICE)

predictor = SAM2ImagePredictor(sam2_model)

# Generate image embedding
predictor.set_image(image_rgb)

# Print predictor size for debugging
print(predictor._features["image_embed"].shape, predictor._features["image_embed"][-1].shape)

# Predict the mask for the bounding box
masks, scores, logits = predictor.predict(
    point_coords=input_point, 
    point_labels=input_label,
    multimask_output = True
)

# Sort the masks by score, in descending order
sorted_ind = np.argsort(scores)[::-1]
masks = masks[sorted_ind]
scores = scores[sorted_ind]
logits = logits[sorted_ind]

# plot results
show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)