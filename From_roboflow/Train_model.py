# Fine tuning of YOLO

# This code allows to leverage the dataset downlaoded with 'Import_labelled_data.py' to train a YOLO 
# model some epochs and the results are stored in the 'results' variable. 
# The model is saved in the folder 'runs'

from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolov8n.pt")

# With absolute path (to be avoided, but it works)
#results = model.train(data = 'C:/Users/chris/OneDrive - Politecnico di Milano/Politecnico di Milano/PhD - dottorato/GitHub repositories Lenovo/Regrasping_balordo/Train-bottle-detection-1/data.yaml', epochs = 30, imgsz = 640, device = 'cpu')

# With relative path (preferred, to be tested)
results = model.train(data = str(Path("Train-bottle-detection-1/data.yaml")), epochs = 30, imgsz = 640, device = 'cpu')