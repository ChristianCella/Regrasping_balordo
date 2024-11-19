# Fine tuning of YOLO

# This code allows to leverage the dataset downlaoded with 'Import_labelled_data.py' to train a YOLO 
# model for some epochs and the results are stored in the 'results' variable. 
# The model is saved in the folder 'runs' and the weights in the corresponding folder.

from ultralytics import YOLO
from pathlib import Path
import torch

verbose = False

if __name__ == '__main__':

    # Print some checks, if necessary
    if verbose:
        print("CUDA available:", torch.cuda.is_available())
        print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

    # Create an instance of the YOLO model
    model = YOLO("yolov8n.pt")
    results = model.train(data = str(Path("WARA_bottles_detection-1/data.yaml")), epochs = 200, imgsz = 640, workers = 8, batch = 32, device = 0)
