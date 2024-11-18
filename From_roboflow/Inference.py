# This code allows to upload the model obtained with 'Train_model.py' to make inference

from ultralytics import YOLO
from pathlib import Path

#model = YOLO("C:/Users/chris/OneDrive - Politecnico di Milano/Politecnico di Milano/PhD - dottorato/GitHub repositories Lenovo/Regrasping_balordo/runs/detect/train9/weights/best.pt")
model = YOLO(str(Path("runs/detect/train9/weights/best.pt")))
image_path = Path("Images_on_slider/Bottle11.png")
results = model.predict(str(image_path), imgsz = 640, conf = 0.2, save = True)