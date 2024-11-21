# This code allows to take a dataset of images and create bounding boxes around the objects in the images. 
# The code saves the images with the bounding boxes and also creates label files in YOLO format for the detected objects. 
# The code can be used to create a labelled dataset for object detection tasks.
# TODO: Implement a way to create 3 different folders: for training, validation and testing datasets.

import os
import cv2
from PIL import Image
from pathlib import Path
from ultralytics import YOLO

IMAGE_FOLDER = str(Path("Images/Augmented_dataset"))
OUTPUT_FOLDER = str(Path("Images/Labelled"))
FIXED_LABEL = "bottle"

# Create the model
model = YOLO("yolov8n.pt")

# Ensure output folders exist
os.makedirs(os.path.join(OUTPUT_FOLDER, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "labels"), exist_ok=True)

# Loop through images in the folder
for image_path in Path(IMAGE_FOLDER).glob("*.jpg"):

    # Read image using OpenCV
    img = cv2.imread(str(image_path))
    img_height, img_width = img.shape[:2]
    
    # Run detection
    results = model(image_path)
    detections = results[0].boxes  # Access the first result's bounding boxes
    
    if len(detections) > 0:  # If any objects are detected

        # Save image to the output folder
        output_image_path = os.path.join(OUTPUT_FOLDER, "images", image_path.name)
        
        # Create label file
        label_file = os.path.join(OUTPUT_FOLDER, "labels", f"{image_path.stem}.txt")

        with open(label_file, "w") as f:
            for box in detections:
                # Extract YOLO outputs
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())  # Bounding box coordinates
                confidence = box.conf[0]                 # Confidence score
                class_id = int(box.cls[0])               # Class ID
                
                # Normalize bounding box coordinates for YOLO format
                x_center = ((x_min + x_max) / 2) / img_width
                y_center = ((y_min + y_max) / 2) / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height
                
                # Write to label file in YOLO format
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                # Draw the bounding box on the image (red color, thickness 2)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                # Optionally, add the label
                label = f"{class_id} {confidence:.2f}"
                cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Save the image with bounding boxes
        cv2.imwrite(output_image_path, img)
