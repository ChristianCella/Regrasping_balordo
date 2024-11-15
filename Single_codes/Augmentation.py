from keras.src.legacy.preprocessing.image import ImageDataGenerator
from skimage import io
from pathlib import Path

datagen = ImageDataGenerator(
    rotation_range = 20,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0,
    zoom_range = 0,
    horizontal_flip = True,
    fill_mode = 'nearest')

# Good heterogeneous images: 1, 2, 7 
image_path = Path("Raw_images/Bottle/XY_plane/x_y_plane_4_Color.png")
img = io.imread(image_path)

img = img.reshape((1,) + img.shape)

i = 0
for batch in datagen.flow(img, batch_size = 16, save_to_dir = 'Augmented_XY_bottles', save_prefix = 'aug', save_format = 'png'):
    i += 1
    if i > 40:
        break
