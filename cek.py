import os
train_files = os.listdir("train")

train_files[19]

#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

import numpy as np
from imageio import imread

import warnings
from scipy import misc

print("selesai 5\n")
#np.savez("train", X=train_data_x, Y=train_data_y)
#np.savez("valid", X=valid_data_x, Y=valid_data_y)
import numpy as np 

import warnings
warnings.filterwarnings('ignore')
import matplotlib.image as mpimg
from skimage.transform import resize
path_to_image = "1.jpg"
custom_image = np.asarray(mpimg.imread(path_to_image))
custom_image_resized = resize(custom_image, (64, 64, 3))
custom_image_resized = np.expand_dims(custom_image_resized, axis=0)
custom_image_resized = image2vec(custom_image_resized) / 255.

print("selesai 7\n")