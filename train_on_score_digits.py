import math
import numpy as np
import matplotlib.pyplot as plt

# from skimage import data
from skimage import transform as tf

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

flattened_images = np.loadtxt("flattened_images.txt")#,  fmt='%i')
classifications = np.loadtxt("classifications.txt") #, , fmt='%i')


images = [ np.reshape(z, (RESIZED_IMAGE_HEIGHT, -1)) for z in flattened_images ]

tform = tf.SimilarityTransform(scale=1, rotation=math.pi/20,
                               translation=(1, -10))


print(classifications[0], tf.warp(images[0], tform))

rotated = tf.warp(images[0], tform)
back_rotated = tf.warp(rotated, tform.inverse)

fig, ax = plt.subplots(nrows=3)

ax[0].imshow(images[0], cmap=plt.cm.gray)
ax[1].imshow(rotated, cmap=plt.cm.gray)
ax[2].imshow(back_rotated, cmap=plt.cm.gray)

for a in ax:
    a.axis('off')

plt.tight_layout()