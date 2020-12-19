import numpy as np
import cv2
import matplotlib.pyplot as plt

def AWB(img_path):
    original_web = plt.imread(img_path)
    rgb_image = cv2.normalize(original_web.astype('float'),None,0.0,1.0,cv2.NORM_MINMAX)

    II = np.zeros(shape=rgb_image.shape,dtype=np.float32)
    
    ave_r = np.mean(rgb_image[:, :, 0])
    print(ave_r)
    ave_g = np.mean(rgb_image[:, :, 1])
    ave_b = np.mean(rgb_image[:, :, 2])
    ave = np.mean([ave_r,ave_g,ave_b])


    
    II[:, :, 0] = (0.5/ave_r) * rgb_image[:, :, 0]
    II[:, :, 1] = (0.5/ave_g) * rgb_image[:, :, 1]
    II[:, :, 2] = (0.5/ave_b) * rgb_image[:, :, 2]
   
    
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_image)     #Red
    plt.title('Orginal Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(II)
    plt.title('After Grey-World Image')
    plt.axis('off')
    plt.show()

    return II
 

if __name__ == '__main__':
    img_path = 'D:/UvA_courses/Computer_Vision/Lab1/lab1/colour_constancy/awb.jpg'
    AWB(img_path )