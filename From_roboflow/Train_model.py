# Fine tuning of YOLO

# This code allows to leverage the dataset downlaoded with 'Import_labelled_data.py' to train a YOLO 
# model some epochs and the results are stored in the 'results' variable. 
# The model is saved in the folder 'runs'

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(data = 'C:/Users/chris/OneDrive - Politecnico di Milano/Politecnico di Milano/PhD - dottorato/GitHub repositories Lenovo/Regrasping_balordo/Train-bottle-detection-1/data.yaml', epochs = 30, imgsz = 640, device = 'cpu')