# This code allows to upload the model obtained with 'Train_model.py' to make inference

from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt

model = YOLO(str(Path("Training/train/weights/best.pt")))
image_path = Path("Images/Random_validation_images/20241127_135349.jpg")
save_directory = Path("Images/Detection_results")

# Inference
results = model.predict(
    str(image_path), 
    imgsz = 640, 
    conf = 0.6, 
    save = False) 

# Result of the detection
detected_image = results[0].plot()

# Save the image
save_path = save_directory / f"{image_path.stem}_predictions.jpg"
plt.imsave(str(save_path), detected_image)

# Display the image
plt.imshow(detected_image)
plt.axis('off')  # Turn off axis for better visualization
plt.title("Detected becker")
plt.show()
