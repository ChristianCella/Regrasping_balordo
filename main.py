# --------------- Simple script for the detection stage

# Import the YOLO class from the ultralytics module
from ultralytics import YOLO

# Load the model and the weights
model = YOLO('yolo11n.pt')

# Perform object detection on an image 
results = model("/home/christian/projects/Regrasping_balordo/Images/Bottle1.png")
results[0].show()
