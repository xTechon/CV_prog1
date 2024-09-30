import time
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import scipy.ndimage as ndi



# function for applying a gaussian blur on an RGB image
def gaussainBlur(src):
    # define a 5x5 gaussian kernel
    gaussian = np.array([[ 1, 4, 6, 4, 1 ],
                         [ 4, 16, 24, 16, 4 ],
                         [ 6, 24, 36, 24, 6],
                         [ 4, 16, 24, 16, 4],
                         [ 1, 4, 6, 4, 1]
                        ])
    # normalize the kernel
    gaussian = gaussian / 256

    # split into channels
    ch1 = src[:, :, 0]
    ch2 = src[:, :, 1]
    ch3 = src[:, :, 2]

    # convolve image and gaussian kernel
    out1 = ndi.convolve(ch1, gaussian, mode='nearest')
    out2 = ndi.convolve(ch2, gaussian, mode='nearest')
    out3 = ndi.convolve(ch3, gaussian, mode='nearest')

    # recombine the image RGB channels
    output = np.dstack([out1, out2, out3])
    return output

def reduceImg(src):
    # initalize an output image at half size of source
    output = src[::2, ::2, :]
    return output

# load image
face1 = iio.imread('./face256.png')

working_img = face1

# smooth image
out1 = gaussainBlur(working_img)

# reduce image
out2 = reduceImg(out1)

# Plotting setup

figure = plt.figure(figsize=(10,7))
rows = 1
columns = 3

figure.add_subplot(rows, columns, 1)
plt.imshow(face1)
plt.title("Original Image")

figure.add_subplot(rows, columns, 2)
plt.imshow(out1)
plt.title("Gaussian Blur")

figure.add_subplot(rows, columns, 3)
plt.imshow(out2)
plt.title("Reduce Image")

plt.show()
