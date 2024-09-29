import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import time

# setup figure plot
figure = plt.figure(figsize=(10,7))
rows = 2
columns = 3

affineTransform = np.float32([[1, 1, 4],
                              [1, 3, 0],
                              [0, 0, 1]])

shear = np.float32([[1, 2, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

root = (np.sqrt(2)/2)

# 45 degree rotation
rotation = np.float32([[root, -root, 0],
                       [root, root, 0],
                       [0, 0, 1]])

sq = (np.sqrt(2))
magnification = np.float32([[sq, 0, 0],
                            [0, sq, 0],
                            [0, 0, 1]])

translation = np.float32([[1, 0, 4],
                          [0, 1, 0],
                          [0, 0, 1]])

i1 = iio.imread('blackSquare.png')
i2 = iio.imread('face.png')
i3 = iio.imread('rectangle.png')

# for every pixel in the defined output image size
# inverse transform onto the original image
# get the average value of the nearest pixels around the original image
# interpolate if needed
def bilinearInterpolation(src, inverse, y, x):
    # dimensions of source image
    height, width, _ = src.shape

    # setup the coordinate vector from output image
    coord = np.array([x, y, 1])

    # find the point in the initial image
    xi, yi, _ = (inverse @ coord)

    # setup coordinates of pixels
    x1 = np.floor(xi).astype(int)
    x2 = np.ceil(xi).astype(int)
    y1 = np.floor(yi).astype(int)
    y2 = np.ceil(yi).astype(int)

    # return 0 pixel value if coordinates out of bounds of initial image
    if x2 >= width or x1 < 0 or y2 >= height or y1 < 0:
        #print("out of bounds")
        return (np.array([0,0,0]).astype(np.uint8))

    # get the originial surrounding values
    p11 = src[y1, x1, :]
    p12 = src[y2, x1, :]
    p21 = src[y1, x2, :]
    p22 = src[y2, x2, :]

    # return if landed on valid initial coord
    if (x1 == x2) and (y1 == y2):
        return src[int(y1), int(x1), :]

    # only one of the dimensions is a vaild initial coord
    elif (x1 == x2):
        value = p11 * ((y2 - yi)/(y2 - y1)) + p12 * ((yi - y1)/(y2 - y1))
        return value
    elif (y1 == y2):
        value = p12 * ((x2 - xi)/(x2 - x1)) + p22 * ((xi - x1)/(x2 - x1))
        return value

    # weighted mean problem
    row1 = p11 * ((x2 - xi)/(x2 - x1)) + p21 * ((xi - x1)/(x2 - x1))
    row2 = p12 * ((x2 - xi)/(x2 - x1)) + p22 * ((xi - x1)/(x2 - x1))
    value = row1 * ((y2 - yi)/(y2 - y1)) + row2 * ((yi - y1)/(y2 - y1))

    return value

# transform image given a image in src, a 2x3 matrix, and an output size
# will use some kind of interpolation
def imgTransform(src, matrix, outputSize=None):
    # calculate smallest possible size needed
    height, width, _ = src.shape
    width, height, _ = (matrix @ [width-1, height-1, 1]).astype(int)
    #print ("height ", height, ", width ", width, ", area", height * width)

    # fallback if rotation causes 0,0 for max height/width and is undefined
    if height == 0 or width == 0:
        height = 2 * src.shape[1]
        width = 2 * src.shape[0]
    # set output size manually if set
    if outputSize is not None:
        height = outputSize[0]
        width = outputSize[1]

    # init output values
    output = np.zeros((height, width, 3), dtype=np.uint8)

    # take inverse of matrix
    inverse = np.linalg.inv(matrix)

    # itterate rows
    for y, row in enumerate(output):
        # itterate columns
        for x, pixel in enumerate(row):
            # Apply to new image
            output[y, x, :] = bilinearInterpolation(src, inverse, y, x)
    return output

start_time = time.time()

working_img = i1
print(working_img.shape)
print("shear")
out1 = imgTransform(working_img, shear)
print("rotation")
out2 = imgTransform(working_img, rotation, (72, 35))
print("magnification")
out3 = imgTransform(working_img, magnification)
print("translation")
out4 = imgTransform(working_img, translation)
print("composition")
out5 = imgTransform(working_img, affineTransform)
print(time.time() - start_time, " seconds")

# add image to plot

figure.add_subplot(rows, columns, 1)

plt.imshow(working_img)
plt.title("Original Image")

figure.add_subplot(rows, columns, 2)
plt.imshow(out1)
plt.title("Shear")

figure.add_subplot(rows, columns, 3)
plt.imshow(out2)
plt.title("Rotation (about origin)")

figure.add_subplot(rows, columns, 4)
plt.imshow(out3)
plt.title("Magnification")

figure.add_subplot(rows, columns, 5)
plt.imshow(out4)
plt.title("Translation")

figure.add_subplot(rows, columns, 6)
plt.imshow(out5)
plt.title("Composition of Previous Transforms")

plt.show()
