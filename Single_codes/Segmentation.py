from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the weights allowing for segmentation
# yolov8n-seg is a smaller model; use yolov8m-seg or yolov8l-seg for larger models
model = YOLO("yolov8l-seg.pt")  

# Define a function to perform segmentation on an input image
def segment_image(image_path):
    # Load the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for plotting
    img_height, img_width = img.shape[:2]

    # Run segmentation
    results = model(img)
    # Process each result (usually there's one result for an image)
    for result in results:
        # Get the masks and resize them to the original image dimensions
        masks = result.masks.data  # Segmentation masks as a tensor
        for mask in masks:
            # Resize the mask to the original image size
            mask_resized = cv2.resize(mask.cpu().numpy(), (img_width, img_height))
            
            # Apply the mask on the image (adjust alpha for visibility)
            img_rgb[mask_resized > 0.5] = [0, 255, 0]  # Example: Set mask areas to green

    # Show the original and segmented images side by side
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Segmented Image")
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()

# Example usage (try for example: Raw_images/Bottle/XY_plane/x_y_plane_1_Color.png)
segment_image("Augmented_XY_bottles/aug_0_1056.png")
