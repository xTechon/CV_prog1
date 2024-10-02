import copy
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import cv2 as cv

# define a 5x5 gaussian kernel
GKERNEL = np.array([[ 1, 4, 6, 4, 1 ],
                         [ 4, 16, 24, 16, 4 ],
                         [ 6, 24, 36, 24, 6],
                         [ 4, 16, 24, 16, 4],
                         [ 1, 4, 6, 4, 1]
                        ])
# normalize the kernel
GKERNEL = GKERNEL / 256

# add plot to figure
def addPlot(figure, row, column, position, img, title=""):
    figure.add_subplot(rows, columns, position) 
    plt.imshow(img)
    plt.title(title)

# creates a composite image with original image on left and pyramid layers on right
def compositeImage(pyramid):
    # original image dimensions
    height, width, channels = pyramid[0].shape

    # determine the size of the composite image
    composite_height = max(height, sum(layer.shape[0] for layer in pyramid[1:]))
    composite_width = width + pyramid[1].shape[1]

    # initalize composite image
    output = np.zeros((composite_height, composite_width, channels), dtype=np.uint8)

    # focus the original image on the left
    output[:height, :width, :] = pyramid[0]

    img_offset = 0
    for layer in pyramid[1:]:
        layer_height, layer_width = layer.shape[:2]
        output[img_offset : img_offset + layer_height, width : width + layer_width] = layer
        img_offset += layer_height

    return output

# blur based on a kernel
def gaussianBlur(src, mult=1):
    # scale the guassian by a factor mult
    gaussian = GKERNEL * mult

    # split into channels
    ch1 = src[:, :, 0]
    ch2 = src[:, :, 1]
    ch3 = src[:, :, 2]

    # convolve image and gaussian kernel
    ch1 = ndi.convolve(ch1, gaussian, mode='constant')
    ch2 = ndi.convolve(ch2, gaussian, mode='constant')
    ch3 = ndi.convolve(ch3, gaussian, mode='constant')

    # recombine the image RGB channels
    output = cv.merge((ch1, ch2, ch3))
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

# reduces an image by half
def reduceImg(src):
    # get every other y, x, pixel value
    output = src[::2, ::2, :]
    return output

# create an image's gaussian and laplacian pyramid
def imgPyramids(src, levels=4):

    # create datastructure
    output = { "gaussian": [], "laplacian": []}

    # base layer of gaussian pyramid
    output["gaussian"].append(copy.deepcopy(src))

    # inital image
    current = src

    # generate pyramid
    for _ in range(levels):
        # new layer
        gaussian = current

        # smooth and reduce image
        gaussian = gaussianBlur(gaussian)
        gaussian = reduceImg(gaussian)

        # make sure image it's all ints
        gaussian = np.rint(gaussian)
        gaussian = np.array(gaussian, dtype=np.uint8)

        # add new gaussian layer to pyramid
        output["gaussian"].append(copy.deepcopy(gaussian))

        # laplace layer upsample
        laplace = upsample(gaussian)
        # Gk - Gk+1
        laplace = current - laplace

        # make sure it's all ints
        laplace = np.rint(laplace)
        laplace = np.array(laplace, dtype=np.uint8)

        # add laplace layer to pyramid
        output["laplacian"].append(copy.deepcopy(laplace))

        # itterate for next loop
        current = gaussian
    return output

# blend the pyramids of two images and a Black and white mask
def blendPyramids(white, black, mask):
    # init output
    output = {"gaussian": [], "laplacian": []}

    # construct base layer
    A = white["gaussian"][-1]
    B = black["gaussian"][-1]
    M = mask["gaussian"][-1] / 255
    base = A * M + (1 - M) * B
    
    # make sure base is all ints
    base = np.rint(base)
    base = np.array(base, dtype=np.uint8)

    output["gaussian"].append(copy.deepcopy(base))

    # blend laplacian layers
    for layer, m in reversed(list(enumerate(mask["gaussian"][:-1]))):

        # split the image channels
        whiteChannels = cv.split(white["laplacian"][layer])
        blackChannels = cv.split(black["laplacian"][layer])
        maskChannels = cv.split((m / 255))

        laplaceChannels = []
        for channel, mask in enumerate(maskChannels):
            a = whiteChannels[channel]
            b = blackChannels[channel]
            out = a * mask + (1-mask) * b

            # make sure out is ints
            out = np.rint(out)
            out = np.array(out, dtype=np.uint8)
            laplaceChannels.append(out)
        
        # merge the channels
        laplace = cv.merge(laplaceChannels)

        # add to pyramid
        output["laplacian"].append(copy.deepcopy(laplace))
    
    # reverse order for consistency
    output["laplacian"] = output["laplacian"][::-1]

    return output




# Reconstruct an image from it's pyramids
def collapsePyramid(pyramid):
    # get the top layer of gaussian pyramid
    current = pyramid["gaussian"][-1]

    for laplace in reversed(pyramid["laplacian"]):
        # upsample the current layer
        current = upsample(current)

        # add the missing details
        current = current + laplace

    # make sure current is an np int array
    current = np.rint(current)
    current = np.array(current, dtype=np.uint8)
    return current

#################
# MAIN FUNCTION
#################


face1 = iio.imread('./face256.png')
face2 = iio.imread('./2face256.png')
mask = iio.imread('./2Mask256.png')

face1 = np.array(face1, dtype=np.uint8)
face2 = np.array(face2, dtype=np.uint8)
mask = np.array(mask, dtype=np.uint8)

face1_pyramid = imgPyramids(face1)
face2_pyramid = imgPyramids(face2)
mask_pyramid = imgPyramids(mask)

blended_pyramid = blendPyramids(face2_pyramid, face1_pyramid, mask_pyramid)
composite = compositeImage(blended_pyramid["laplacian"])

result = collapsePyramid(blended_pyramid)


# plotting setup
figure = plt.figure(figsize=(10,7))
rows = 1
columns = 5

addPlot(figure, rows, columns, 1, face1, "face 1")
addPlot(figure, rows, columns, 2, face2, "face 2")
addPlot(figure, rows, columns, 3, mask, "mask")
addPlot(figure, rows, columns, 4, result, "result")

plt.show()