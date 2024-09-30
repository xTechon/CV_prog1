import copy
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
#import cv2 as cv



# function for applying a gaussian blur on an RGB image
def gaussianBlur(src, mult=1):
    # define a 5x5 gaussian kernel
    gaussian = np.array([[ 1, 4, 6, 4, 1 ],
                         [ 4, 16, 24, 16, 4 ],
                         [ 6, 24, 36, 24, 6],
                         [ 4, 16, 24, 16, 4],
                         [ 1, 4, 6, 4, 1]
                        ])
    # normalize the kernel
    gaussian = gaussian / 256

    # scale the guassian by a factor mult
    gaussian = gaussian * mult

    # split into channels
    ch1 = src[:, :, 0]
    ch2 = src[:, :, 1]
    ch3 = src[:, :, 2]

    # convolve image and gaussian kernel
    ch1 = ndi.convolve(ch1, gaussian, mode='constant')
    ch2 = ndi.convolve(ch2, gaussian, mode='constant')
    ch3 = ndi.convolve(ch3, gaussian, mode='constant')

    # recombine the image RGB channels
    output = np.dstack([ch1, ch2, ch3])
    return output

# reduces an image by half
def reduceImg(src):
    # get every other y, x, pixel value
    output = src[::2, ::2, :]
    return output

# upsample image by 2
def upsample(src):
    height, width, _ = src.shape
    output = np.zeros((2*height, 2*width, 3), dtype=np.uint8)
    
    # upsample
    output[::2, ::2, :] = src
    # blur image
    output = gaussianBlur(output, 4)
    return output


# creates a the image pyramids of levels from an image src
def imgPyramids(src, levels=3):
    
    output = {"gaussian": [], "laplacian": []}
    output["gaussian"].append(src)

    current = src
    for _ in range(levels):
        # set the next layer
        temp = current

        # Generate next Gaussian layer
        temp = gaussianBlur(temp)
        temp = reduceImg(temp)
        output["gaussian"].append(copy.copy(temp))

        # Generate next Laplacian layer
        difference = upsample(temp)
        difference = current - difference
        output["laplacian"].append(copy.copy(difference))

        current = temp        
    return output

# creates a composite image with original image on left and pyramid layers on right
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

# create pyramids
out = imgPyramids(working_img, 4)

# create composite images
out1 = compositeImage(out["gaussian"])
out2 = compositeImage(out["laplacian"])

# Plotting setup
figure = plt.figure(figsize=(10,7))
rows = 1
columns = 2

"""
figure.add_subplot(rows, columns, 1)
plt.imshow(working_img)
plt.title("Original Image")
"""

figure.add_subplot(rows, columns, 1)
plt.imshow(out1)
plt.title("Gaussian Pyramid")

figure.add_subplot(rows, columns, 2)
plt.imshow(out2)
plt.title("Laplaccian Pyramid")

plt.show()
