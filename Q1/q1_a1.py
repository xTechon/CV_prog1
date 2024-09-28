from scipy import misc
from scipy import ndimage
import numpy as np
import imageio
f = misc.face()

affineTransform = np.array([[1, 1],
                            [1, 3]])
translation = np.array([[4],
                        [0]])

inverseTransform = np.linalg.inv(affineTransform)

i = imageio.imread('blackSquare.png')
#imageio.imsave('facerat.JPG', f) # uses the Image module (PIL)

print(f)

#out = ndimage.affine_transform(i, inverseTransform)
#out =


import matplotlib.pyplot as plt
plt.imshow(i)
#plt.imshow(out)
plt.show()
