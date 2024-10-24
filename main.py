# Solve import problems
import sys
sys.path.append("/home/christian/projects/Regrasping_balordo/ultralytics")

# Import YOLO class
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt") # Pretrained model
# model = YOLO("/home/christian/projects/Regrasping_balordo/ultralytics/runs/detect/train/weights/best.pt") # Custom weights

'''
# Train the model
train_results = model.train(
    data = "coco8.yaml",  # path to dataset YAML
    epochs = 100,  # number of training epochs
    imgsz = 640,  # training image size
    device = "cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()
'''

# Perform object detection on an image
results = model("/home/christian/projects/Regrasping_balordo/Images/Bottle11.png")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model