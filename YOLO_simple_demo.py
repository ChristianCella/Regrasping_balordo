from pathlib import Path
from ultralytics import YOLO

# Load the model and the weights fro segmentation
model = YOLO("yolo11n-seg.pt")

# Perform object detection on an image 
image_path = Path("Images/Bottle1.png")
results = model(image_path)
results[0].show()
