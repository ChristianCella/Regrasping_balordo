# Import necessary libraries
from ultralytics import YOLO
from pathlib import Path
import cv2 
import matplotlib.pyplot as plt
import matplotlib

# Data for LaTeX font
matplotlib.rcParams['mathtext.fontset'] = 'cm' # 'cm' or 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rc('xtick', labelsize = 15)
matplotlib.rc('ytick', labelsize = 15)

# Define the variables
verbose = 1 # if verbiage is 1, then print the results
bottle_class_id = 39 # The class ID for bottles

# Load the model and the weights (do not push them on GitHub!!)
model = YOLO('yolo11n.pt')
#model = YOLO('yolov8x-worldv2.pt')

# Define the image path
image_path = Path("Images_on_slider/Bottle1.png")

# Load the image
image = cv2.imread(str(image_path))
image_final = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

# If needed, display the image before modifications
if verbose == 1:
    image_rgb = image.copy()
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
    plt.figure(figsize = (10, 10))
    plt.title("Original Image", fontsize = 20)
    plt.imshow(image_rgb)
    plt.axis('off')  # Hide axis for a cleaner look
    plt.show()

# Perform object detection on the image
results = model(image_path)

# Display all the results
if verbose == 1:
    plt.figure(figsize = (10, 10))
    plt.title("Complete detection", fontsize = 20)
    plt.imshow(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide axis for a cleaner look
    plt.show()

# If needed, print some checks
if verbose == 1:
    for box in results[0].boxes:

        # Get some properties of the detected object
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        class_id = int(box.cls[0])  # Class ID of the detected object
        confidence = box.conf[0]  # Confidence score

        # Display the detected object's class ID, bounding box coordinates, and confidence score
        print(f"Detected object with class ID {class_id} at [{x1}, {y1}, {x2}, {y2}] with confidence {confidence}")

# Loop through each object that was detected (if any)
for result in results[0].boxes:

    # Get the class ID of the detected object
    class_id = int(result.cls[0])  

    # Check if the detected object is a bottle
    if class_id == bottle_class_id:

        # Get the bounding box coordinates
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Convert coordinates to integers

        # Draw the bounding box on the image
        cv2.rectangle(image_final, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for bottles

# Display the image with the bounding box around the detected bottle
plt.figure(figsize = (10, 10))
plt.title("Detected bottle", fontsize = 20)
plt.imshow(image_final)
plt.axis('off')  # Hide axis for a cleaner look
plt.show()