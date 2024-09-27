import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import pprint


figure = plt.figure(figsize=(10,7))

rows = 1
columns = 2

affineTransform = np.float32([[1, 1, 4],
                            [1, 3, 0],
                            [0, 0, 1]])
transformTest = np.array([[1, 0, 4],
                            [0, 1, 0],
                            [0, 0, 1]])

i1 = iio.imread('blackSquare.png')
#i2 = iio.imread('face.jpg')

# transform image given a image in src, a 2x3 matrix, and an output size
# will use some kind of interpolation
def imgTransform(src, matrix, outputSize):
    # init output values
    output = np.zeros((outputSize[1], outputSize[0], 3), dtype=np.uint8)

    # itterate rows
    for x, row in enumerate(src):
        # itterate columns
        for y, pixel in enumerate(row):
            # coordinate in homogenous
            coord = np.array([x, y, 1])
            # output coordinates
            x_out, y_out, _ = (matrix @ coord).astype(np.uint8)
            # print (coord, x_out, y_out, pixel)
            # Apply to new image
            output[x_out, y_out, :] = pixel
    return output


out1 = imgTransform(i1, transformTest, (200,100))

figure.add_subplot(rows, columns, 1)

plt.imshow(i1)
plt.title("before Transform")

figure.add_subplot(rows, columns, 2)
plt.imshow(out1)
plt.title("after Transform")

plt.show()
