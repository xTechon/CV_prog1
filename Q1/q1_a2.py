import cv2 as cv
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt


figure = plt.figure(figsize=(10,7))

rows = 1
columns = 2

affineTransform = np.float32([[1, 1, 4],
                            [1, 3, 0]])

i1 = iio.imread('blackSquare.png')
i2 = iio.imread('face.jpg')

out1 = cv.warpAffine(i1, affineTransform, (125, 200), cv.INTER_LINEAR)

figure.add_subplot(rows, columns, 1)

plt.imshow(i1)
plt.title("before Transform")

figure.add_subplot(rows, columns, 2)
plt.imshow(out1)
plt.title("after Transform")

plt.show()
