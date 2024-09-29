# Programming Assignment 1
William Daniel Hiromoto
Code can be found at [Github Repo](https://github.com/xTechon/CV_prog1/tree/main)

----------------------------------------------
## Program 1 - Affine Map

We have the following affine map:

$$
    \left [
    \begin{matrix}
        x1 \\ x2
    \end{matrix}
    \right ]
    \to
    \left [
    \begin{matrix}
        1 & 1 \\
        1 & 3
    \end{matrix}
    \right ]
    \left [
    \begin{matrix}
        x1 \\ x2
    \end{matrix}
    \right ]
    +
    \left[
    \begin{matrix}
        4 \\ 0      
    \end{matrix}
    \right ]
$$

This affine map can be converted into homogenous coordnates in 3 Dimensions to include the translation:

$$
    \left [
    \begin{matrix}
        1 & 1 & 4\\
        1 & 3 & 0\\
        0 & 0 & 1
    \end{matrix}
    \right ]
$$

This can then be decomposed into the following transformation matricies:

Shear: $\left [
    \begin{matrix}
        1 & 2 & 0\\
        0 & 1 & 0\\
        0 & 0 & 1
    \end{matrix}
    \right ]$
Rotation: $\left [
    \begin{matrix}
        \frac{\sqrt2}{2} & -\frac{\sqrt2}{2} & 0\\
        \frac{\sqrt2}{2} & \frac{\sqrt2}{2} & 0\\
        0 & 0 & 1
    \end{matrix}
    \right ]$
Scale: $\left [
    \begin{matrix}
        \sqrt2 & 0 & 0\\
        0 & \sqrt2 & 0\\
        0 & 0 & 1
    \end{matrix}
    \right ]$
Translation: $\left [
    \begin{matrix}
        1 & 0 & 4\\
        0 & 1 & 0\\
        0 & 0 & 1
    \end{matrix}
    \right ]$

In order to apply these transformation to an image in python, we'll first import a few external libraries, namely, `numpy`, `imageio`, and `matplotlib.pyplot`

After loading an image of a black square on a white background, we need to itterate over every pixel of this image. Before we can do so, we need to understand how images are stored once read by the imageio library.

```python       
i1 = iio.imread('blackSquare.png')
```

![50 x 50 Black Square on White Background](Q1/blackSquareBorder.png "Black Square"){style="border: 1px solid black"}

Instead of every pixel value having a coordinate, it is placed in an array, with the "Y-Coordinates" Stored first followed by the "X-Coordinates". This data structure will be a 3D array, as the color data (RGB) of every pixel is stored in an array of length 3. For simplicity sake, I will only be working with images that do not have an alpha/transparency channel:
```
[
    [[255, 255, 255], [255, 255, 255], ..., [255, 255, 255]],
    [[0, 0, 0],       [0, 0, 0],       ..., [0, 0, 0]],
    [[255, 255, 255], [255, 255, 255], ..., [255, 255, 255]],
]
```
\centerline{Data Structure of images when read}

The "Coordinate" values will corresspond to indexes in the image array. I.e. `i1[y, x, :]` will get the pixel data at coordinate x,y. Finally, the origin begins at the top left of the image, and the positive Y coordinates go to the bottom of the image and positive X coordinates go the right of the image.


