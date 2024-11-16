# In this code you can use the model trained in roboflow without physically importing the files
# in the working directory
# Look at the other codes to understand how to train locally

# Import libraries
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from roboflow import Roboflow
from pathlib import Path
import cv2
import matplotlib

# Data for LaTeX font
matplotlib.rcParams['mathtext.fontset'] = 'cm' # 'cm' or 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rc('xtick', labelsize = 15)
matplotlib.rc('ytick', labelsize = 15)

# Initialize Roboflow with the key API
rf = Roboflow(api_key = "ieuVIgoSLJvfjf8y5cco")

# Access the project and model (specify the correct version)
project = rf.workspace().project("train-bottle-detection")
model = project.version(1).model

# Define the image path
image_path = Path("Images_on_slider/Bottle2.png")

# Perform inference (filter out predictions with confidence below 40% and overlap below 30%)
response = model.predict(str(image_path), confidence = 40, overlap = 30).json()

# Load the image using OpenCV
image = cv2.imread(str(image_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Matplotlib

# Extract predictions
predictions = response.get("predictions", [])

# Plot the image with bounding boxes
fig, ax = plt.subplots(1, figsize = (12, 8))
ax.imshow(image)

# Add bounding boxes and labels
for pred in predictions:

    # Get the bounding box properties
    x, y, width, height = pred["x"], pred["y"], pred["width"], pred["height"]
    class_name = pred["class"]
    confidence = pred["confidence"]

    # Calculate the top-left corner for Matplotlib (x_min, y_min)
    top_left = (x - width / 2, y - height / 2)

    # Add a rectangle patch for the bounding box
    rect = patches.Rectangle(
        top_left, width, height, linewidth = 2, edgecolor = "red", facecolor = "none"
    )
    ax.add_patch(rect)

    # Add the class label and confidence
    label = f"{class_name} ({confidence:.2f})"
    ax.text(
        x - width / 2,
        y - height / 2 - 10,
        label,
        color = "black",
        fontsize = 10,
        bbox = dict(facecolor = "white", alpha = 0.5),
    )

# Hide axes and show the plot
ax.axis("off")
plt.title('Detected objects', fontsize = 20)
plt.show()
