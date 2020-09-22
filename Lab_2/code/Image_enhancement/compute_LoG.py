import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage, misc, signal
from gauss2D import gauss2D

def compute_LoG(image, LOG_type):
    if LOG_type == 1:
        imOut = cv2.GaussianBlur(image,(5,5),0.5)
        imout = cv2.Laplacian(imOut,cv2.CV_64F)
    elif LOG_type == 2:
        imout = scipy.ndimage.filters.gaussian_laplace(image,0.5)
    elif LOG_type == 3:
        f_gauss1 = gauss2D(10,5)
        f_gauss2 = gauss2D(0.5,5)
        imout = scipy.signal.convolve2d(image, f_gauss1 - f_gauss2)
    return imout

if __name__ == '__main__':
    default_path = 'D:/UvA_courses/Computer_Vision/lab2_new/Image_enhancement/images/image2.jpg'
    #Read two images with float!
    orig_image = plt.imread(default_path)
    orig_image = orig_image.astype(np.float64)
    
    #The parameter is denoise(image,type of filter,kernel demision,kernel demision for median, sigma for gaussian) 
    out1 = compute_LoG(orig_image,1)
    out2 = compute_LoG(orig_image,2)
    out3 = compute_LoG(orig_image,3)

    
    #Show images in greyscale
    plt.figure ()
    plt.subplot(141)
    plt.imshow(orig_image, cmap='gray')
    plt.title('orig_image')
    plt.axis("off")
    
    plt.subplot(142)
    plt.imshow(out1, cmap='gray')
    plt.title('Method 1')
    plt.axis("off")
    
    plt.subplot(143)
    plt.imshow(out2, cmap='gray')
    plt.title('Method 2')
    plt.axis("off")
    
    plt.subplot(144)
    plt.imshow(out3, cmap='gray')
    plt.title('Method 3')
    plt.axis("off")
    plt.show()

