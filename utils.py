import numpy as np
import matplotlib.pyplot as plt

# Useful functions for visualization

def show_mask(mask, ax, random_color=False, borders = True):
    """
    Visualizes a segmentation mask on a given matplotlib axis.
    Args:
        mask (np.array): The binary mask to display.
        ax (matplotlib axis): The axis on which to plot the mask.
        random_color (bool): Whether to use a random color for the mask.
        borders (bool): Whether to draw borders around the mask regions.
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    """
    Displays positive and negative points on the image.
    Args:
        coords (np.array): Coordinates of points (Nx2).
        labels (np.array): Labels for the points (1 for positive, 0 for negative).
        ax (matplotlib axis): The axis on which to plot the points.
        marker_size (int): Size of the marker.
    """
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    """
    Visualizes a bounding box on the image.
    Args:
        box (list or np.array): Bounding box coordinates [x_min, y_min, x_max, y_max].
        ax (matplotlib axis): The axis on which to plot the box.
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    """
    Visualizes segmentation masks, scores, and optionally input points and bounding boxes.
    Args:
        image (np.array): The image to overlay masks on.
        masks (list of np.array): List of binary masks to display.
        scores (list of float): Scores associated with each mask.
        point_coords (np.array, optional): Coordinates of input points.
        box_coords (np.array, optional): Bounding box coordinates.
        input_labels (np.array, optional): Labels for input points (positive/negative).
        borders (bool): Whether to draw borders around the mask regions.
    """
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