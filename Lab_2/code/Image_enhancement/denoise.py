import numpy as np
import  cv2
import matplotlib.pyplot as plt
from myPSNR import myPSNR 

def denoise( image, kernel_type, *kwargs):
    if kernel_type == 'box':
        imOut = cv2.blur(image, kwargs[0]);
    elif kernel_type == 'median':
        print(kwargs[0])
        imOut = cv2.medianBlur(image, kwargs[1])
    elif kernel_type == 'gaussian':
        imOut = cv2.GaussianBlur(image,kwargs[0],kwargs[2])
    else:
        print('Operatio Not implemented')
    return imOut
    
    
if __name__ == '__main__':
    default_path = 'D:/UvA_courses/Computer_Vision/lab2_new/Image_enhancement/images/image1.jpg'
    gaussian_path = 'D:/UvA_courses/Computer_Vision/lab2_new/Image_enhancement/images/image1_gaussian.jpg'
    saltpepper_path = 'D:/UvA_courses/Computer_Vision/lab2_new/Image_enhancement/images/image1_saltpepper.jpg'
    
    image = plt.imread(gaussian_path)
    
    #The parameter is denoise(image,type of filter,kernel demision,kernel demision for median, sigma for gaussian) 
    out_3 = denoise(image,'gaussian',(3,3),3,0.5)
    out_5 = denoise(image,'gaussian',(3,3),5,1)
    out_7 = denoise(image,'gaussian',(3,3),7,2)
    myPSNR(image, out_3)
    myPSNR(image, out_5)
    myPSNR(image, out_7)
    
    #Show images in greyscale
    plt.figure()
    plt.subplot(141)
    plt.title('original image')
    plt.imshow(image, cmap='gray')
    plt.axis("off")
    
    plt.subplot(142)
    plt.title('Sigma = 0.5')
    plt.imshow(out_3, cmap='gray')
    plt.axis("off")
    plt.subplot(143)
    plt.title('Sigma = 1.0')
    plt.imshow(out_5, cmap='gray')
    plt.axis("off")
    plt.subplot(144)
    plt.title('Sigma = 2.0')
    plt.imshow(out_7, cmap='gray')
    plt.axis("off") 
    plt.show()