import copy
import time
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import scipy.ndimage as ndi



# function for applying a gaussian blur on an RGB image
def gaussianBlur(src):
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

# reduces an image by half
def reduceImg(src):
    # get every other y, x, pixel value
    output = src[::2, ::2, :]
    return output

# creates a gaussian pyramid of leves from an image src
def gaussianPyramid(src, levels=2):
    output = []
    output.append(src)

    temp = src
    for x in range(levels):

        temp = gaussianBlur(temp)
        temp = reduceImg(temp)
        output.append(copy.copy(temp))

    return output

def compositeImage(pyramid):
    # original image dimensions
    height, width, _ = pyramid[0].shape

    # determine the size of the composite image
    composite_height = max(height, sum(layer.shape[0] for layer in pyramid[1:]))
    composite_width = width + pyramid[1].shape[1]
    output = np.zeros((composite_height, composite_width, 3), dtype=np.uint8)

    # focus the original image on the left
    output[:height, :width, :] = pyramid[0]

    img_offset = 0
    for layer in pyramid[1:]:
        layer_height, layer_width = layer.shape[:2]
        output[img_offset : img_offset + layer_height, width : width + layer_width] = layer
        img_offset += layer_height

    return output

# load image
face1 = iio.imread('./face512.png')

working_img = face1

out = gaussianPyramid(working_img, 4)
out1 = compositeImage(out)

# Plotting setup
"""
figure = plt.figure(figsize=(10,7))
rows = 1
columns = 3

figure.add_subplot(rows, columns, 1)
"""
plt.imshow(out1)
plt.title("Gaussian Pyramid")

plt.show()
