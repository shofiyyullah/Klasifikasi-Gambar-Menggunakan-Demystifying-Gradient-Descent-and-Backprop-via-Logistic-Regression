import os
train_files = os.listdir("train")

train_files[19]

#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

fig=plt.figure(figsize=(20, 20))
for i in range(1, 15):
    img_name = train_files[random.randint(0, len(train_files))]
    img=mpimg.imread('train/'+img_name)
    #fig.add_subplot(5, 5, i)
    #plt.imshow(img)
#plt.show()

print("selesai 1\n")

import numpy as np
train_size = int(len(train_files) * 0.8)
train_data_x = np.zeros((train_size, 64, 64, 3))
train_data_y = np.zeros((1, train_size))
valid_data_x = np.zeros((len(train_files) - train_size, 64, 64, 3))
valid_data_y = np.zeros((1, len(train_files) - train_size))

print("selesai 2\n")

from imageio import imread
train_data = []
for i, im in enumerate(train_files):
    filename = "train/" + im
    image = np.asarray(imread(filename))
    train_data.append((image, 1 if im.split(".")[0] == "cat" else 0))
    print(i)

print("selesai 3\n")

c = 0
for x,y in train_data:
    c += x.shape[0] >= 64 and x.shape[1] >= 64
    print(x)
c

print("selesai 4\n")

import warnings
from scipy import misc
warnings.simplefilter("ignore", DeprecationWarning)
for i, (x, y) in enumerate(train_data):
    resized_image = misc.imresize(x, (64, 64, 3))
    print(i)
    if i < train_size:
        train_data_x[i] = resized_image
        train_data_y[:, i] = y
    else:
        valid_data_x[i - train_size] = resized_image
        valid_data_y[:, i - train_size] = y
print("selesai 5\n")

np.savez("train.npz", X=train_data_x, Y=train_data_y)
np.savez("valid.npz", X=valid_data_x, Y=valid_data_y)
print("selesai 6 npz!\n")