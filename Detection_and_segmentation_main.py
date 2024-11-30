import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import sys
import cv2

# Add the 'sam2' folder to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sam2_path = os.path.join(script_dir, "sam2")
scripts_path = os.path.join(script_dir, "scripts")
sys.path.append(sam2_path)
sys.path.append(scripts_path)

# Import utilities and modules for object detection, segmentation, and visualization
from scripts import fonts
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ultralytics import YOLO
from utils import show_points, show_box, show_masks


# Define paths to the input image and pre-trained model checkpoints
input_rgb_path = "/Project/Regrasping_balordo/data/labelled_bins/images/bins/flash/20241128_100416.png"
SAM_checkpoint_path = "/Project/Regrasping_balordo/sam2/checkpoints/sam2.1_hiera_large.pt"
YOLO_checkpoint_path = "/Project/Regrasping_balordo/YOLO_tests/detect/train3/weights/best.pt"


# Select the computation device based on availability (CUDA, MPS, or CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

# Enable specific optimizations for CUDA devices if applicable
if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


if __name__ == "__main__":

    # Perform object detection using the YOLO model
    print(f'{fonts.green}Performing object detection with YOLO! :){fonts.reset}')
    YOLO_model = YOLO(YOLO_checkpoint_path)
    results = YOLO_model.predict(input_rgb_path, show=True)
    
    # Extract bounding boxes, labels, and probabilities from YOLO predictions
    boxes = results[0].boxes.data.cpu().numpy()[:,:4]
    labels = results[0].boxes.cls.cpu().numpy()
    probabilities = results[0].boxes.data.cpu().numpy()[:,-2]
    classes_dictionary = results[0].names

    # Find the object with the highest detection probability
    max_prob_idx = np.argmax(probabilities)
    max_prob_box = boxes[max_prob_idx]
    max_prob_label = labels[max_prob_idx]
    max_prob_class = classes_dictionary[max_prob_label]

    # Print details of the detected object
    print(f'{fonts.purple}The detected object with the highest probability is a', max_prob_class, f'{fonts.reset}')
    print(f'{fonts.purple_light}The box coordinates are:', max_prob_box, f'{fonts.reset}')
    print(f'{fonts.purple_light}The probability is:', probabilities[max_prob_idx], f'{fonts.reset}')

    # Define input for segmentation (point and bounding box)
    x1, y1, x2, y2 = max_prob_box[0], max_prob_box[1], max_prob_box[2], max_prob_box[3]
    x_center, y_center  = (x1 + x2) / 2, (y1 + y2) / 2 
    input_box = max_prob_box
    input_label = np.array([int(max_prob_label)])
    input_point = np.array([[int(x_center),int(y_center)]])

    print(f'{fonts.red_light}The SAM input point is:', x_center,',', y_center , f'{fonts.reset}')
    
    # Load the SAM2 segmentation model
    print(f'{fonts.red_light}Loading the SAM2 model...{fonts.reset}')
    sam2_checkpoint = SAM_checkpoint_path
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    # Load the input image for processing
    print(f'{fonts.red_light}Loading the image...{fonts.reset}')
    image = cv2.imread(input_rgb_path)
    print(f'{fonts.red_light}Enhancing the image...{fonts.reset}')
    # Enhance the image using CLAHE and other techniques to improve contrast
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # Split the LAB image into separate channels (L, A, B)
    l_channel, a_channel, b_channel = cv2.split(lab)
    # Apply CLAHE to the L-channel to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
    l_channel = clahe.apply(l_channel)
    # (Optional) Apply histogram equalization to further stretch contrast
    l_channel = cv2.equalizeHist(l_channel)
    # Merge the enhanced L-channel back with the original A and B channels
    lab = cv2.merge((l_channel, a_channel, b_channel))
    # Convert the LAB image back to the BGR color space
    enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # (Optional) Further increase contrast using convertScaleAbs
    alpha = 1.5  # Contrast control
    beta = 0     # Brightness control
    enhanced_image = cv2.convertScaleAbs(enhanced_image, alpha=alpha, beta=beta)
    # (Optional) Apply Unsharp Masking for edge enhancement
    blurred = cv2.GaussianBlur(enhanced_image, (0, 0), 3)
    enhanced_image = cv2.addWeighted(enhanced_image, 1.5, blurred, -0.5, 0)
    
    # Save the enhanced image to a new file
    input_directory = os.path.dirname(input_rgb_path)
    input_filename = os.path.basename(input_rgb_path)
    output_filename = f"enhanced_{input_filename}"
    output_path = os.path.join(input_directory, output_filename)
    cv2.imwrite(output_path, enhanced_image)
    print(f'{fonts.red_light}The enhanced image has been saved as:', output_path, f'{fonts.reset}')
    print(f'{fonts.yellow}oh wow, This is the object to segment! Close the image and I will predict! {fonts.reset}')

    # Visualize the detected bounding box and input point on the original image
    predictor.set_image(image)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    show_box(input_box, plt.gca())
    plt.axis('on')
    plt.show()  
    print(f'{fonts.green}Predicting the SEGMENTAION masks! :) {fonts.reset}')

    # Perform segmentation using the SAM2 model
    masks, scores, logits = predictor.predict(
    point_coords = input_point,
    point_labels = input_label,
    box = input_box,
    multimask_output=True,
    )
    
    # Display the segmentation results
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    show_masks(image, [masks[0]], [scores[0]], point_coords=input_point, input_labels=input_label, borders=True)
    print(f'{fonts.green}The end! :D {fonts.reset}')