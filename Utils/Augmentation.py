# This code allows to manually augment the images in the dataset. 
# The code uses the ImageDataGenerator class from Keras to perform the augmentation. 
# The code reads an image from a specified path (manually specify it), reshapes it, and applies the augmentation transformations. 
# In the end, the images should be saved in a temporary folder, so that they can be checked before 
# being included in the dataset.


from keras.src.legacy.preprocessing.image import ImageDataGenerator
from skimage import io
from pathlib import Path

# Define how many 'synthetic' image should be generated
num_img = 20

# Define the augmentation transformations
datagen = ImageDataGenerator(
    rotation_range = 20,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0,
    zoom_range = 0,
    horizontal_flip = True,
    fill_mode = 'nearest')

# Good heterogeneous images: 1, 2, 7 (for the bottles)
image_path = Path("Raw_images/Bottle/Bottle bottom/Bottom_4_Color.png")
img = io.imread(image_path)

# Reshape the image
img = img.reshape((1,) + img.shape)

# Perform the augmentation until you generate the desired number of images
i = 0
for batch in datagen.flow(img, batch_size = 16, save_to_dir = 'Attempt_folder', save_prefix = 'aug', save_format = 'png'):
    i += 1
    if i > num_img:
        break
