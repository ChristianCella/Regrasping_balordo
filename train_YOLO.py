from ultralytics import YOLO
from pathlib import Path
import torch

if __name__ == '__main__':
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")



    #model = YOLO("/Project/Regrasping_balordo_taffi/weights_christian/best.pt")
    # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
    #results = model.predict(source="/Project/Regrasping_balordo_taffi/data/images/bins/flash", show=True,save=True) 

    # Create an instance of the YOLO model
    # model = YOLO("yolo11n.pt")
    # results = model.train(data = str(Path("/Project/Regrasping_balordo_taffi/data/dataset_ready_YOLO/data.yaml")), epochs = 100, imgsz = 640, workers = 8, batch = 32, device = 0)

    # # Evaluate model performance on the validation set
    # metrics = model.val()

    # # Perform object detection on an image
    # model.predict("/Project/Regrasping_balordo_taffi/data/images/6_items/", show=True)
