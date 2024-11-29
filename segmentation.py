from pathlib import Path
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from math import atan2, cos, sin, sqrt, pi

from YOLO_simple_demo import YOLO_detect

from segment_anything_2.sam2.build_sam import build_sam2
from segment_anything_2.sam2.sam2_image_predictor import SAM2ImagePredictor

########################## UTILITY FUNCTIONS ############################
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

def getOrientation(pts, img):
    
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
 
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
 
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    
 
    
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 255, 0), 5)
 
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    
 
    return angle

def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
 
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
 
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
 
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)


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

# Turn the points outside the mask black
mask = masks[0]
mask = mask.astype(np.uint8)
mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
mask = mask.astype(np.bool)
image[mask == False] = 0
image[mask == True] = 255

# Plot the final image
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
plt.title('Segmented image BW mask')
plt.show()

# Save the final image
cv2.imwrite('segmented_image.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

print('Segmentation completed!')

# Convert mask to binary
image_gr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# plot image in grayscale
plt.figure(figsize=(10, 10))
plt.imshow(image_gr, cmap='gray')
plt.axis('off')
plt.title('Grayscale image')
plt.show()

# Apply thresholding to the mask
# _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# plot image
# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# plt.axis('off')
# plt.show()

############# PCA ################
# Find all the contours in the thresholded image
contours, _ = cv2.findContours(image_gr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
 
for i, c in enumerate(contours):
    # Calculate the area of each contour
    area = cv2.contourArea(c)
    # Ignore contours that are too small or too large
    if area < 1e2 or 1e5 < area:
        continue
 
    # Draw each contour only for visualisation purposes
    cv2.drawContours(image_gr, contours, i, (0, 0, 255), 2)
    # Find the orientation of each shape through PCA
    getOrientation(c, image)

# plot image
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
plt.show()