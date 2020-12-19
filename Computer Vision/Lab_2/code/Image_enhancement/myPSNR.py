import numpy as np
import  cv2
import matplotlib.pyplot as plt

def myPSNR( orig_image, approx_image ):
    #Calculate the MSE
    squared_diff = (orig_image -approx_image) ** 2
    summed = np.sum(squared_diff)
    num_pix = approx_image.shape[0] * approx_image.shape[1] #img1 and 2 should have same shape
    MSE = summed / num_pix
    #print("MSE IS: ", np.max(orig_image))
    
    #Calculate the PSNR
    PSNR = 10 * np.log10(np.max(orig_image)**2/(MSE))
    print("PSNR IS:",PSNR)
    return PSNR

    
if __name__ == '__main__':
    img1_path = 'D:/UvA_courses/Computer_Vision/lab2_new/Image_enhancement/images/image1.jpg'
    img2_path = 'D:/UvA_courses/Computer_Vision/lab2_new/Image_enhancement/images/image1_gaussian.jpg'
    img3_path = 'D:/UvA_courses/Computer_Vision/lab2_new/Image_enhancement/images/image1_saltpepper.jpg'
    
    #Read two images with float!
    orig_image = plt.imread(img1_path)
    orig_image = cv2.normalize(orig_image.astype('float64'),None,0.0,1.0,cv2.NORM_MINMAX)
    
    approx_image = plt.imread(img2_path)
    approx_image = cv2.normalize(approx_image.astype('float64'),None,0.0,1.0,cv2.NORM_MINMAX)
    
    myPSNR(orig_image,approx_image)