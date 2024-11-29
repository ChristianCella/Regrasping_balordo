import os
import shutil
import random

# Set the directories for images and labels
current_directory = os.path.dirname(os.path.abspath(__file__))
# IMAGE_FOLDER = os.path.join(current_directory, 'data', 'images')
# LABEL_FOLDER = os.path.join(current_directory, 'data', 'labels')
IMAGE_FOLDER = os.path.join('/Project/Regrasping_balordo/data/labelled_bottles/valid', 'images')
LABEL_FOLDER = os.path.join('/Project/Regrasping_balordo/data/labelled_bottles/valid', 'labels')
TRAIN_FOLDER = os.path.join(current_directory, 'data', 'dataset_ready_YOLO', 'train')
VAL_FOLDER = os.path.join(current_directory, 'data', 'dataset_ready_YOLO', 'val')
TEST_FOLDER = os.path.join(current_directory, 'data', 'dataset_ready_YOLO', 'test')

# Ensure the destination folders exist
os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(VAL_FOLDER, exist_ok=True)
os.makedirs(TEST_FOLDER, exist_ok=True)

def split_data(image_folder, label_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Ensure the sum of the ratios is 1
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("The sum of train_ratio, val_ratio, and test_ratio must be 1.")

    # Get the list of image files (excluding hidden files)
    image_files = []
    for root, _, files in os.walk(image_folder):
        for file in files:
            # Check if the file is an image (you can adjust the extensions as needed)
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_files.append(os.path.join(root, file))    
    # Shuffle the image files to randomize the split
    random.shuffle(image_files)

    # Calculate the split indices
    total_files = len(image_files)
    train_size = int(total_files * train_ratio)
    val_size = int(total_files * val_ratio)
    
    # Split the data into train, validation, and test sets
    train_files = image_files[:train_size]
    val_files = image_files[train_size:train_size + val_size]
    test_files = image_files[train_size + val_size:]

    # Function to move files to their respective folder
    def move_files(file_list, src_folder, dst_folder):
        for file in file_list:
            image_path = os.path.join(src_folder, file)
            label_path = os.path.join(LABEL_FOLDER, os.path.splitext(os.path.basename(file))[0] + '.txt')
            if os.path.exists(image_path):
                shutil.copy(image_path, dst_folder)
            if os.path.exists(label_path):
                shutil.copy(label_path, dst_folder)

    # Move the files
    move_files(train_files, image_folder, TRAIN_FOLDER)
    move_files(val_files, image_folder, VAL_FOLDER)
    move_files(test_files, image_folder, TEST_FOLDER)

    # Print out the result
    print(f"Total images: {total_files}")
    print(f"Training set: {len(train_files)} images")
    print(f"Validation set: {len(val_files)} images")
    print(f"Test set: {len(test_files)} images")


if __name__ == "__main__":
    # Ask user for the split ratios (with default values)
    try:
        train_ratio = float(input(f"Enter training data percentage (default 0.8): ") or 0.8)
        val_ratio = float(input(f"Enter validation data percentage (default 0.1): ") or 0.1)
        test_ratio = float(input(f"Enter test data percentage (default 0.1): ") or 0.1)
        
        # Ensure the sum of the ratios is 1
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
            print("The sum of the ratios must be 1. Using default values instead.")
            train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1

        # Split the data
        split_data(IMAGE_FOLDER, LABEL_FOLDER, train_ratio, val_ratio, test_ratio)

    except ValueError as e:
        print(f"Error: {e}")
