import numpy as np
import cv2
from scipy import signal
import scipy
import matplotlib.pyplot as plt

def compute_gradient(image):
    #define G_x and G_y
    G_x = np.ones(shape = image.shape)
    G_y = np.ones(shape = image.shape)
    
    #Define two kernel matrix
    kernel_x = [[1 , 0 , -1],[2 , 0 ,-1],[1 ,0 ,-1]]
    kernel_y = [[1, 2, 1],[0 , 0, 0],[-1 ,-2, -1]]
    print("image.shape: ", image.shape)
    
    #use convolve2d
    G_x = scipy.signal.convolve2d(image,kernel_x)
    G_y = scipy.signal.convolve2d(image,kernel_y)
    im_magnitude = np.sqrt(G_x ** 2 + G_y ** 2)
    
    im_direction = np.arctan(G_y / G_x)
    
    plt.figure ()
    plt.subplot(221)
    plt.imshow(G_x,cmap='gray')
    plt.title('Gradient in the x-direction')
    plt.subplot(222)
    plt.imshow(G_y,cmap='gray')
    plt.title('Gradient in the y-direction')
    plt.subplot(223) 
    plt.imshow(im_magnitude,cmap='gray')
    plt.title('Gradient magnitude each pixel')
    plt.subplot(224) 
    plt.imshow(im_direction,cmap='gray')
    plt.title('Gradient direction each pixel')
    plt.show()
    return Gx, Gy, im_magnitude,im_direction

if __name__ == '__main__':
    default_path = 'D:/UvA_courses/Computer_Vision/lab2_new/Image_enhancement/images/image2.jpg'
    orig_image = plt.imread(default_path)
    compute_gradient(orig_image)
   

