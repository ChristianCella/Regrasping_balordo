import os
import random
import cv2
from pathlib import Path
from ultralytics import YOLO

# Define constants
IMAGE_FOLDER = str(Path("Images/Augmented_dataset_beckers"))
OUTPUT_FOLDER = str(Path("Images/Labelled"))
#FIXED_LABEL = "becker"
SPLIT_RATIOS = {"train": 0.7, "valid": 0.2, "test": 0.1}

# Assign colors for each split (BGR format for OpenCV)
SPLIT_COLORS = {
    "train": (0, 255, 0),  # Green for train
    "valid": (255, 0, 0),  # Blue for valid
    "test": (0, 0, 255),   # Red for test
}

# Create output folders
for split in SPLIT_RATIOS.keys():
    os.makedirs(os.path.join(OUTPUT_FOLDER, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, split, "labels"), exist_ok=True)

# Create the YOLO model
model = YOLO("yolov8n.pt")

# Get all image paths
image_paths = list(Path(IMAGE_FOLDER).glob("*.jpg"))

# Shuffle the images to randomize the splits
random.shuffle(image_paths)

# Calculate the split indices
num_images = len(image_paths)
train_end = int(SPLIT_RATIOS["train"] * num_images)
valid_end = train_end + int(SPLIT_RATIOS["valid"] * num_images)

# Split the images into train, valid, and test sets
splits = {
    "train": image_paths[:train_end],
    "valid": image_paths[train_end:valid_end],
    "test": image_paths[valid_end:]
}

# Process and save the images
for split, paths in splits.items():
    for image_path in paths:
        # Read image using OpenCV
        img = cv2.imread(str(image_path))
        img_height, img_width = img.shape[:2]
        
        # Run YOLO detection
        results = model(image_path)
        detections = results[0].boxes  # Access the first result's bounding boxes
        
        # Define output paths
        split_image_folder = os.path.join(OUTPUT_FOLDER, split, "images")
        split_label_folder = os.path.join(OUTPUT_FOLDER, split, "labels")
        output_image_path = os.path.join(split_image_folder, image_path.name)
        label_file = os.path.join(split_label_folder, f"{image_path.stem}.txt")
        
        if len(detections) > 0:  # If any objects are detected
            # Find the bounding box with the largest area
            largest_box = None
            largest_area = 0
            
            for box in detections:
                # Extract YOLO outputs
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())  # Bounding box coordinates
                area = (x_max - x_min) * (y_max - y_min)
                
                if area > largest_area:
                    largest_area = area
                    largest_box = box

            if largest_box:
                # Extract coordinates for the largest box
                x_min, y_min, x_max, y_max = map(int, largest_box.xyxy[0].tolist())
                confidence = largest_box.conf[0]                 # Confidence score
                class_id = int(largest_box.cls[0])               # Class ID
                
                # Normalize bounding box coordinates for YOLO format
                x_center = ((x_min + x_max) / 2) / img_width
                y_center = ((y_min + y_max) / 2) / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height
                
                # Draw the largest bounding box on the image
                #color = SPLIT_COLORS[split]  # Get the color for the current split
                #cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
                # Optionally add label text
                #label = f"{class_id} {confidence:.2f}"
                #cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Write to label file in YOLO format
                with open(label_file, "w") as f:
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            # Save the image with the bounding box
            cv2.imwrite(output_image_path, img)
