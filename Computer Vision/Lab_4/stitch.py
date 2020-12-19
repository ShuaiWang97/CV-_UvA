import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
MIN = 10
starttime=time.time()
img1 = cv2.imread('./left.jpg') #query
img2 = cv2.imread('./right.jpg') #train



surf=cv2.SIFT_create()
kp1,descrip1=surf.detectAndCompute(img1,None)
kp2,descrip2=surf.detectAndCompute(img2,None)
FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
searchParams = dict(checks=50)

flann=cv2.FlannBasedMatcher(indexParams,searchParams)
match=flann.knnMatch(descrip1,descrip2,k=2)


good=[]
for i,(m,n) in enumerate(match):
        if(m.distance<0.7*n.distance):
                good.append(m)

if len(good)>MIN:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        ano_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        M,mask=cv2.findHomography(src_pts,ano_pts,cv2.RANSAC,5.0)
        warpImg = cv2.warpPerspective(img2, np.linalg.inv(M), (img1.shape[1]+img2.shape[1], img1.shape[0]))
        direct=warpImg.copy()
        direct[0:img1.shape[0], 0:img1.shape[1]] =img1
        
        final=time.time()
        img3=cv2.cvtColor(direct,cv2.COLOR_BGR2RGB)
        
        
        
        
        #show result
        plt.subplot(1,2,1)
        plt.imshow(img1[:,:,::-1])
        plt.title("Left")
        plt.axis("off")
        plt.subplot(1,2,2)
        plt.imshow(img2[:,:,::-1])
        plt.title("Right")
        plt.axis("off")
        plt.show()
        cv2.waitKey()
        plt.imshow(img3,)
        plt.axis("off")
        plt.show()
        cv2.waitKey()

        cv2.imwrite("simplepanorma.png",direct)
        cv2.imwrite("bestpanorma.png",warpImg)
        
else:
        print("not enough matches!")