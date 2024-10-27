from ultralytics import YOLO

# Load the model
model = YOLO('yolo11n.pt')

# Perform object detection on an image and specify save directory
results = model("/home/christian/projects/Regrasping_balordo/Images/Bottle1.png")
results[0].show()
