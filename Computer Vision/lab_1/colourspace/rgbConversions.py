import numpy as np
import cv2

import math

def rgb2grays(input_image):
    # converts an RGB into grayscale by using 4 different methods

    # ligtness method
    shape = input_image.shape
    new_image = np.empty(shape)

    for x, y in np.ndindex(shape[0], shape[1]):
        RGB = input_image[x,y]

        pix = (max(RGB) - min(RGB))/2

        new_image[x, y] = pix

    return np.uint8(new_image)

    # average method
    # shape = input_image.shape
    # new_image = np.empty(shape)
    #
    # for x, y in np.ndindex(shape[0], shape[1]):
    #     [R, G, B] = input_image[x,y]
    #
    #     pix = (R + G + B)/3
    #
    #     new_image[x, y] = pix
    #
    # return np.uint8(new_image)

    # luminosity method - WIP
    # shape = input_image.shape
    # new_image = np.empty(shape)
    #
    # for x, y in np.ndindex(shape[0], shape[1]):
    #     [R, G, B] = input_image[x,y]
    #
    #     pix = 0.21*R + 0.72*G + 0.07*B
    #
    #     new_image[x, y] = pix
    #
    # return np.uint8(new_image)

    # built-in opencv function

    # return cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)




def rgb2opponent(input_image):
    # converts an RGB image into opponent colour space
    shape = input_image.shape
    new_image = np.empty(shape)

    for x, y in np.ndindex(shape[0], shape[1]):
        [R, G, B] = input_image[x,y]

        O1 = (R - G)/math.sqrt(2)
        O2 = (R + G - 2*B)/math.sqrt(6)
        O3 = (R + G + B)/math.sqrt(3)

        new_image[x, y] = np.array([O1, O2, O3])

    return np.uint8(new_image)



def rgb2normedrgb(input_image):
    # converts an RGB image into normalized rgb colour space
    shape = input_image.shape
    new_image = np.empty(shape)

    for x, y in np.ndindex(shape[0], shape[1]):
        [R, G, B] = input_image[x,y]

        r = R/(R + G + B)
        g = G/(R + G + B)
        b = B/(R + G + B)

        new_image[x, y] = np.array([r, g, b])

    return new_image
