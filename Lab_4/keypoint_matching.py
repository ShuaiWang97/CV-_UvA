import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage, misc, signal
from matplotlib.patches import ConnectionPatch
from skimage.transform import warp


def keypoint_matching(image1, image2):
    
    #use SIFT algorithm
    sift = cv2.SIFT_create()
    print(sift)
    kp1, des1 = sift.detectAndCompute(image1,None)
    kp2, des2 = sift.detectAndCompute(image2,None)
    
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)
    plt.show()

    # Match descriptors.
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x: x.distance)
    img3 = cv2.drawMatches(image1,kp1,image2,kp2,matches[:10],None, flags=2)
    plt.imshow(img3)
    plt.axis('off')
    plt.title('random subset(10) of all matching points')
    plt.show()
    
    #visualize the result
    plt.figure(figsize=(15,15))
    img1 = cv2.drawKeypoints(image1,kp1,image1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2 = cv2.drawKeypoints(image2,kp2,image2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    '''
    plt.subplot(1,2,1)
    plt.imshow(img1)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(img2)
    plt.axis('off')
    plt.show()
    '''


if __name__ == '__main__':
    path1 = './boat1.pgm'
    path2 = './boat2.pgm'
    image1 = cv2.imread(path1,0)
    image2 = cv2.imread(path2,0)
    keypoint_matching(image1,image2)