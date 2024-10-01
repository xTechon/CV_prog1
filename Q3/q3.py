import copy
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import cv2 as cv

# creates a composite image with original image on left and pyramid layers on right
def compositeImage(pyramid):
    # original image dimensions
    height, width = pyramid[0].shape

    # determine the size of the composite image
    composite_height = max(height, sum(layer.shape[0] for layer in pyramid[1:]))
    composite_width = width + pyramid[1].shape[1]
    output = np.zeros((composite_height, composite_width), dtype=np.uint8)

    # focus the original image on the left
    output[:height, :width] = pyramid[0]

    img_offset = 0
    for layer in pyramid[1:]:
        layer_height, layer_width = layer.shape[:2]
        output[img_offset : img_offset + layer_height, width : width + layer_width] = layer
        img_offset += layer_height

    return output

# function for applying a gaussian blur on an RGB image channel
def gaussianBlur(channel, mult=1):
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

    # convolve image channel and gaussian kernel
    output = ndi.convolve(channel, gaussian, mode='constant')

    return output

# reduces an image by half
def reduceImg(src):
    # get every other y, x, pixel value
    output = src[::2, ::2]
    return output

# upsample image channel by 2
def upsample(src):
    height, width = src.shape
    output = np.zeros((2*height, 2*width), dtype=np.uint8)
    
    # upsample
    output[::2, ::2] = src[:, :]
    # blur image
    output = gaussianBlur(output * 4)
    return output


# creates a the image channel pyramids of levels from an image src
def imgPyramids(src, levels=3):
    
    output = {"gaussian": [], "laplacian": []}
    output["gaussian"].append(src)

    current = src
    for _ in range(levels):
        # set the next layer
        gaussian = current

        # Generate next Gaussian layer
        gaussian = gaussianBlur(gaussian)
        gaussian = reduceImg(gaussian)

        output["gaussian"].append(copy.copy(gaussian))

        # Generate next Laplacian layer
        laplacian = upsample(gaussian)
        laplacian = current - laplacian

        output["laplacian"].append(copy.copy(laplacian))

        current = gaussian
    return output

# where src is an image pyramid
def reconstructImg(src):
    
    #rows = 1
    #columns = 4
    current = src["gaussian"][-1]
    for layer in reversed(src["laplacian"]):
        #print(current.shape)
        temp = upsample(current)
        
        """
        figure1 = plt.figure(figsize=(10,7), clear=True)
        figure1.add_subplot(rows, columns, 1)
        plt.imshow(current)
        plt.title("current")
        figure1.add_subplot(rows, columns, 2)
        plt.imshow(copy.copy(temp))
        plt.title("upsampled")
        """

        temp = temp + layer

        """
        figure1.add_subplot(rows, columns, 3)
        plt.imshow(layer)
        plt.title("laplace layer")
        figure1.add_subplot(rows, columns, 4)
        plt.imshow(temp)
        plt.title("Addition")
        plt.show()
        """

        current = temp
    return current


# lowest lv layer of both images
# the laplacian pyramid of both images
# gaussian pyramid + 1 size of images
# each input is a pyramid "pair"
def pyramidBlend(pyramidA, pyramidB, mask):

    # create a new image pyramid
    output = {"gaussian": [], "laplacian": []}
    
    base = mask.get("gaussian")[-1] / 255

    # get the lowest layer of mask
    current = (pyramidA["gaussian"][-1] * (base)) + ((1 - base) * pyramidB["gaussian"][-1])
    
    # current = np.zeros((base.shape[0], base.shape[1]))
    
    output["gaussian"].append(current)
    # mask * A + (1-mask) * B
    for layer, m in reversed(list(enumerate(mask["gaussian"][:-1]))):
        """
        figure2 = plt.figure(figsize=(10,7), clear=True)
        rows = 1
        columns = 4
        figure2.add_subplot(rows, columns, 1)
        plt.imshow(pyramidA["laplacian"][layer])
        plt.title("laplacian A")
        figure2.add_subplot(rows, columns, 2)
        plt.imshow(pyramidB["laplacian"][layer])
        plt.title("laplacian B")
        figure2.add_subplot(rows, columns, 3)
        plt.imshow(m)
        plt.title("mask")
        """

        l = (pyramidA["laplacian"][layer] * (m/255)) + ((1-(m/255)) * pyramidB["laplacian"][layer]) 
        """
        figure2.add_subplot(rows, columns, 4)
        plt.imshow(l)
        plt.title("blended layer")
        plt.show()
        """

        # Reconstruct into blended pyramid
        #temp = upsample(current)
        #temp = temp + l
        output["laplacian"].append(l)

    # reverse order for consistency
    output["laplacian"] = output["laplacian"][::-1]
    
    return output

def imageBlend(imgW, imgB, mask, layers=4):

    # split the images
    iwR, iwG, iwB = cv.split(imgW)
    ibR, ibG, ibB = cv.split(imgB)
    m, _, _ = cv.split(mask)

    # create pyramid for the mask
    pM = imgPyramids(m, layers)
    
    # create pyramids for the channels
    pwR = imgPyramids(iwR, layers)
    pwG = imgPyramids(iwG, layers)
    pwB = imgPyramids(iwB, layers)

    pbR = imgPyramids(ibR, layers)
    pbG = imgPyramids(ibG, layers)
    pbB = imgPyramids(ibB, layers)

    # blend the pyramids
    bR = pyramidBlend(pwR, pbR, pM)
    bG = pyramidBlend(pwG, pbG, pM)
    bB = pyramidBlend(pwB, pbB, pM)

    # reconstruct each pyramid
    outR = reconstructImg(bR)
    outG = reconstructImg(bG)
    outB = reconstructImg(bB)

    """
    figure2 = plt.figure(figsize=(10,7), clear=True)
    rows = 1
    columns = 4
    figure2.add_subplot(rows, columns, 1)
    plt.imshow(outR)
    plt.title("Red")
    figure2.add_subplot(rows, columns, 2)
    plt.imshow(outG)
    plt.title("Green")
    figure2.add_subplot(rows, columns, 3)
    plt.imshow(outB)
    plt.title("Blue")
    """
    outR[outR < 0] = 0
    outR[outR > 255] = 255
    outR = outR.astype(np.uint8)
    outG[outG < 0] = 0
    outG[outG > 255] = 255
    outG = outG.astype(np.uint8)
    outB[outB < 0] = 0
    outB[outB > 255] = 255
    outB = outB.astype(np.uint8)

    output = cv.merge((outR, outG, outB))

    """
    figure2.add_subplot(rows, columns, 4)
    plt.imshow(output)
    plt.title("output")
    plt.show()
    """
    return output



# load image
face1 = iio.imread('./face256.png')
fruit1 = iio.imread('./2face256.png')
mask1 = iio.imread('./2Mask256.png')

working_img1 = fruit1
working_img2 = face1
working_img3 = mask1

out1 = imageBlend(working_img1, working_img2, working_img3, 4)

# Plotting setup
figure = plt.figure(figsize=(10,7))
rows = 1
columns = 4


figure.add_subplot(rows, columns, 1)
plt.imshow(working_img1)
plt.title("Operand 1")

figure.add_subplot(rows, columns, 2)
plt.imshow(working_img2)
plt.title("Operand 2")


figure.add_subplot(rows, columns, 3)
plt.imshow(working_img3)
plt.title("Mask")


figure.add_subplot(rows, columns, 4)
plt.imshow(out1)
plt.title("Result")


plt.show()
