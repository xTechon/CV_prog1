import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import pprint
import threading
import concurrent.futures
import time


figure = plt.figure(figsize=(10,7))

rows = 1
columns = 2

affineTransform = np.float32([[1, 1, 4],
                             [1, 3, 0],
                             [0, 0, 1]])

i1 = iio.imread('blackSquare.png')
i2 = iio.imread('face.jpg')


#pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)

# for every pixel in the defined output image size
# inverse transform onto the original image
# get the average value of the nearest pixels around the original image
# interpolate if needed
def bilinearInterpolation(src, matrix, x, y):
    # dimensions of source image
    height, width, _ = src.shape

    # take inverse of matrix
    inverse = np.linalg.inv(matrix)

    # setup the coordinate vector from output image
    coord = np.array([x, y, 1])

    # find the point in the initial image
    xi, yi, _ = (inverse @ coord)

    #print(height, width)
    #print(coord, xi, yi)

    # setup coordinates of pixels
    x1 = np.floor(xi).astype(int)
    x2 = np.ceil(xi).astype(int)
    y1 = np.floor(yi).astype(int)
    y2 = np.ceil(yi).astype(int)

    # return 0 pixel value if coordinates out of bounds of initial image
    if x2 >= width or x1 < 0 or y2 >= height or y1 < 0:
        return (np.array([0,0,0]).astype(np.uint8))

    # return if landed on valid initial coord
    if (x1 == x2) and (y1 == y2):
        #print (int(x1), int(y1), src[int(x1), int(y1), :])
        return src[int(x1), int(y1), :]

    # get the originial surrounding values
    p11 = src[x1, y1, :]
    p12 = src[x1, y2, :]
    p21 = src[x2, y1, :]
    p22 = src[x2, y2, :]

    # weighted mean problem
    row1 = p11 * (x2 - xi) + p21 * (xi - x1)
    row2 = p12 * (x2 - xi) + p22 * (xi - x1)
    value = row1 * (y2 - yi) + row2 * (yi - y1)

    return value

def multithreadHelper(src, matrix, output, x, y):
    output[y, x, :] = bilinearInterpolation(src, matrix, x, y)

# transform image given a image in src, a 2x3 matrix, and an output size
# will use some kind of interpolation
def imgTransform(src, matrix, outputSize=None):
    # calculate smallest possible size needed
    width, height, _ = src.shape
    width, height, _ = (matrix @ [height, width, 1]).astype(int)
    print (height, width, height * width)
    if outputSize is not None:
        height = outputSize[0]
        width = outputSize[1]
    # init output values
    output = np.zeros((height, width, 3), dtype=np.uint8)

    # make multithreaded
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        # itterate rows
        for y, row in enumerate(output):
            # itterate columns
            for x, pixel in enumerate(row):
                # Apply to new image
                #output[y, x, :] = bilinearInterpolation(src, matrix, x, y)
                executor.submit(multithreadHelper, src, matrix, output, x, y)
        print("done submitting workers")
    return output

start_time = time.time()
out1 = imgTransform(i1, affineTransform)
print(time.time() - start_time, " seconds")

# add image to plot

figure.add_subplot(rows, columns, 1)

plt.imshow(i1)
plt.title("before Transform")

figure.add_subplot(rows, columns, 2)
plt.imshow(out1)
plt.title("after Transform")

plt.show()
