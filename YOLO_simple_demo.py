from pathlib import Path
from ultralytics import YOLO

image_path = Path("Images/Bottle1.png")

def YOLO_segment(image_path):
    # Load the model and the weights fro segmentation
    model = YOLO("yolo11n-seg.pt")

    # Perform object detection on an image 
    results = model(image_path)
    results[0].show()
    return results

def YOLO_detect(image_path):
    # Load the model and the weights fro segmentation
    model = YOLO('yolo11n.pt')

    # Perform object detection on an image for class 39, bottle
    results = model.predict(image_path, classes=[39])
    results[0].show()
    return results

# main to execute the function
if __name__ == "__main__":
    pred = YOLO_detect(image_path)

    for r in pred:
        print(r.boxes.cls)
        print(r.boxes)
        
