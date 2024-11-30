from ultralytics import YOLO
from pathlib import Path
import torch

if __name__ == '__main__':
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

    # Create an instance of the YOLO model
    model = YOLO("yolo11n.pt")
    results = model.train(data = str(Path("/Project/Regrasping_balordo/data/dataset_ready_YOLO/data.yaml")), epochs = 300, imgsz = 640, workers = 8, batch = 4, device = 0)

    # # Evaluate model performance on the validation set
    metrics = model.val()

    # # Perform object detection on an image
    model = YOLO("/runs/detect/train3/weights/best.pt")
    model.predict("/Project/Regrasping_balordo/data/images/bins/flash", save=True)
