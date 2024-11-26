# This code allows to upload the model obtained with 'Train_model.py' to make inference

from ultralytics import YOLO
from pathlib import Path


model = YOLO(str(Path("runs/detect/train3/weights/best.pt")))
image_path = Path("Images/Random_validation_images/20241126_160032.jpg")
results = model.predict(str(image_path), imgsz = 640, conf = 0.1, save = True) 
