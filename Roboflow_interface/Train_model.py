# Fine tuning of YOLO

# This code allows to leverage the dataset downlaoded with 'Import_labelled_data.py' to train a YOLO 
# model for some epochs and the results are stored in the 'results' variable. 
# The model is saved in the folder 'runs' and the weights in the corresponding folder.

from ultralytics import YOLO
from pathlib import Path
import torch

verbose = True

if __name__ == '__main__':

    # Print some checks, if necessary
    if verbose:
        print("CUDA available:", torch.cuda.is_available())
        print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

    # Create an instance of the YOLO model and specify a custom directory
    model = YOLO("yolov8n.pt")
    custom_dir = Path("Training")

    # In case of ubuntu (GPU GTX 1650 Ti => muc smaller computational capability)
    results = model.train(
        data = str(Path("Images/Labelled/data.yaml")), 
        epochs = 150, 
        imgsz = 512, 
        workers = 1, 
        batch = 4, 
        device = 0, 
        lr0=1e-4,
        project = str(custom_dir))

    # In case of windows (GPU RTX 4070 ==> much bigger computational capability)    
    #results = model.train(data = str(Path("Images/Labelled/data.yaml")), epochs = 100, imgsz = 640, workers = 8, batch = 32, device = 0, lr0=1e-4, project = str(custom_dir))