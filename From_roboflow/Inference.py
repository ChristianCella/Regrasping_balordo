# This code allows to upload the model obtained with 'Train_model.py' to make inference

from ultralytics import YOLO
from pathlib import Path

model = YOLO("C:/Users/chris/OneDrive - Politecnico di Milano/Politecnico di Milano/PhD - dottorato/GitHub repositories Lenovo/Regrasping_balordo/runs/detect/train9/weights/best.pt")

image_path = Path("Train-bottle-detection-1/valid/images/aug_0_1283_png.rf.6cb95b8aae4bca5aecbff77adb875050.jpg")
results = model.predict(str(image_path), imgsz = 640, conf = 0.2, save = True)