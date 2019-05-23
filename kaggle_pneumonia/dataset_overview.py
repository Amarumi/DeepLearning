import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia#chest_xray.zip

total_images_train_normal = os.listdir('data/train/NORMAL/')
total_images_train_pneumonia = os.listdir('data/train/PNEUMONIA/')

sample_normal = random.sample(total_images_train_normal,6)
f,ax = plt.subplots(2,3,figsize=(15,9))

## ---------------- View example Healthy X-rays ----------------
for i in range(0,6):
    im = plt.imread('data/train/NORMAL/'+sample_normal[i])
    ax[i//3,i%3].imshow(im)
    ax[i//3,i%3].axis('off')
f.suptitle('Normal Lungs')
plt.show()


## --------------- View example Pneumonia X-rays ---------------
sample_pneumonia = random.sample(total_images_train_pneumonia,6)
f,ax = plt.subplots(2,3,figsize=(15,9))

for i in range(0,6):
    im = plt.imread('data/train/PNEUMONIA/'+sample_pneumonia[i])
    ax[i//3,i%3].imshow(im)
    ax[i//3,i%3].axis('off')
f.suptitle('Pneumonia Lungs')
plt.show()


## --------------- View dataset split ---------------
plt.bar(x=['Normal','Pneumonia'],
        height=[len(total_images_train_normal),
                len(total_images_train_pneumonia)],
        color=['lightblue','lightred'])
plt.title('Image volume in train set')
plt.show()

# data/ folder directory structure:
#     train/
#         NORMAL    - 1341 images
#         PNEUMONIA - 3875 images
#     val/
#         NORMAL    - 8 images
#         PNEUMONIA - 8 images
#     test/
#         NORMAL    - 234 images
#         PNEUMONIA - 390 images


