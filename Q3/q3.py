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

# where src is the smallest image and laplacian is the pyramid
def reconstructImg(src, laplacian):
    current = src
    for layer in reversed(laplacian):
        print(current.shape)
        temp = upsample(current)
        temp = temp + layer
        current = temp
    return current


# lowest lv layer of both images
# the laplacian pyramid of both images
# gaussian pyramid + 1 size of images
# each input is a pyramid "pair"
def imageBlend(pyramidA, pyramidB, mask):

    # get the lowest layer of mask
    current = pyramidA["gaussian"][-1]
    # mask * A + (1-mask) * B
    for layer, m in reversed(list(enumerate(mask["gaussian"][:-1]))):
        print(m.shape)
        l = pyramidA["laplacian"][layer] * (m/255) + (1-m / 255) * pyramidB["laplacian"][layer]
        #lg = pyramidA["laplacian"][layer][:,:,1] * m[:,:,1] + (1-m[:,:,1]) * pyramidB["laplacian"][layer][:,:,1]
        #lb = pyramidA["laplacian"][layer][:,:,2] * m[:,:,2] + (1-m[:,:,2]) * pyramidB["laplacian"][layer][:,:,2]

        # Reconstruct into blended image
        temp = upsample(current)
        temp = temp + l
        current = temp
    return current

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
face1 = iio.imread('./face256.png')
fruit1 = iio.imread('./A_Orange256.png')
mask1 = iio.imread('./Mask256.png')

working_img1 = face1
working_img2 = fruit1
working_img3 = mask1

# create pyramids
out1 = imgPyramids(working_img1, 4)
out2 = imgPyramids(working_img2, 4)
out3 = imgPyramids(working_img3, 4)

out4 = imageBlend(out2, out1, out3)
# create composite images
#out1 = compositeImage(out["gaussian"])
#out2 = compositeImage(out2["gaussian"])

#out1 = reconstructImg(out["gaussian"][-1], out["laplacian"])



# Plotting setup
figure = plt.figure(figsize=(10,7))
rows = 1
columns = 4


figure.add_subplot(rows, columns, 1)
plt.imshow(out2["gaussian"][0])
plt.title("Operand 1")

figure.add_subplot(rows, columns, 2)
plt.imshow(out1["gaussian"][0])
plt.title("Operand 2")

figure.add_subplot(rows, columns, 3)
plt.imshow(out3["gaussian"][0])
plt.title("Mask")

figure.add_subplot(rows, columns, 4)
plt.imshow(out4)
plt.title("Result")


plt.show()
