import os

def update_yolo_annotations(directory_path):
    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        # Process only .txt files
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)

            # Read the content of the file
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Modify each line
            updated_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts:  # Ensure the line is not empty
                    parts[0] = '1'  # Replace the first element with '1'
                    updated_lines.append(' '.join(parts))

            # Save the updated content back to the file
            with open(file_path, 'w') as file:
                file.write('\n'.join(updated_lines))
    
    print("Annotation files have been updated successfully.")

# Example usage:
directory_path = '/Project/Regrasping_balordo/data/labelled_bottles/valid/labels/'  # Replace with your directory path
update_yolo_annotations(directory_path)
