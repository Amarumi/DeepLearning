import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from os import mkdir

preview_path = 'preview'

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

img_1 = load_img('data/train/NORMAL/IM-0115-0001.jpeg')  # this is a PIL image
img_2 = load_img('data/train/NORMAL/IM-0166-0001.jpeg')  # this is a PIL image
x = np.stack([img_to_array(img_2), img_to_array(img_2)])  # this is a Numpy array with shape (2, 3, 150, 150)

try:
    # Create target Directory
    mkdir(preview_path)
    print("Directory ", preview_path, " created!")
except FileExistsError:
    print("Directory ", preview_path, " already exists!")

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
for i, batch in enumerate(datagen.flow(x, batch_size=1,
                                       save_to_dir=preview_path,
                                       save_prefix='normal',
                                       save_format='jpeg')):
    if i == 19:
        break  # otherwise the generator would loop indefinitely
