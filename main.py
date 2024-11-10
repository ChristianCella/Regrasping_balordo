# --------------- Simple script for the detection stage

# Import the YOLO class from the ultralytics module
from ultralytics import YOLO
from pathlib import Path

# Load the model and the weights
model = YOLO('yolo11n.pt')

image_path = Path("Images\Bottle1.png")

# Perform object detection on an image 
results = model(image_path)
results[0].show()