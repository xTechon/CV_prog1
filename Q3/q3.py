import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import time

# define the gaussian kernel
gaussian = np.array([[ 1, 4, 6, 4, 1 ],
                     [ 4, 16, 24, 16, 4 ],
                     [ 6, 24, 36, 24, 6],
                     [ 4, 16, 24, 16, 4],
                     [ 1, 4, 6, 4, 1]
                    ])
# normalize the kernel
gaussian = gaussian / 256

# load image
face1 = iio.imread('face256.png')

# split into channels
ch1 = face1[:, :, 0]
ch2 = face1[:, :, 1]
ch3 = face1[:, :, 2]

# convolve image and gaussian kernel

out1 = ndi.convolve(ch1, gaussian, mode='nearest')
out2 = ndi.convolve(ch2, gaussian, mode='nearest')
out3 = ndi.convolve(ch3, gaussian, mode='nearest')

output = np.dstack([out1, out2, out3])

# Plotting setup

figure = plt.figure(figsize=(10,7))
rows = 1
columns = 2

figure.add_subplot(rows, columns, 1)
plt.imshow(face1)
plt.title("Original Image")

figure.add_subplot(rows, columns, 2)
plt.imshow(output)
plt.title("Gaussian Blur")


plt.show()