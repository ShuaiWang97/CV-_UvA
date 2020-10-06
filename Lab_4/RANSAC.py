import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

#can also use the findHomography instead
def perfect_function(src, dst, size):
    A = None
    for idx in range(size):
        x = src[idx][0]
        y = src[idx][1]
        u = dst[idx][0]
        v = dst[idx][1]
        h1 = np.array([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        h2 = np.array([0, 0, 0, -x, -y, -1, v * x, v * y, v])
        if A is None:
            A = np.vstack((h1, h2))
        else:
            A = np.vstack((A, h1))
            A = np.vstack((A, h2))
    u, s, v = np.linalg.svd(A)
    h = np.reshape(v[8], (3, 3))
    h =h * 1 / h.item(8)
    return h
    
def count_inliers(src, dst, H, threshold):
    n_samples = len(src)
    homo_src = np.hstack((src, np.ones((n_samples, 1))))
    homo_dst = np.hstack((dst, np.ones((n_samples, 1))))

    estimate_dst = np.dot(homo_src, H.T)
    estimate_dst[:, 0] /= estimate_dst[:, 2]
    estimate_dst[:, 1] /= estimate_dst[:, 2]
    estimate_dst[:, 2] = 1
    error = (estimate_dst - homo_dst) ** 2
    error = np.sum(error, axis=1)
    error = np.sqrt(error)
    return (error < threshold).sum()

def RANSAC(N, src, dst, threshold):
    n_samples = len(src)
    best_inliers = 0
    best_H = None
    for _ in range(N):
        random_idx = np.random.randint(n_samples, size=4)
        H = perfect_function(src[random_idx], dst[random_idx], size = 4)
        n_inliers = count_inliers(src, dst, H, threshold)
        if (n_inliers > best_inliers):
            best_inliers = n_inliers
            best_H = H
    print('Best number of inliers', best_inliers)
    return best_H

if __name__ == "__main__":
    img2 = cv2.imread('./boat1.pgm')
    img1 = cv2.imread('./boat2.pgm')

    threshold = 5.0
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)

    #good matrix
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append([m])
    #draw the connection
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #plt.imshow(img3)
    plt.show()
    matches = np.asarray(good)
    if (len(matches[:,0]) >= 0):
        src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,2)
        H = RANSAC(100, src, dst, threshold)
    else:
        raise AssertionError('Can’t find enough keypoints.')
    print(H)
    dst = cv2.warpPerspective(img1,H,((img1.shape[1]), img2.shape[0])) #wraped image
    cv2.imwrite('output.jpg',dst)
    plt.imshow(dst)
    plt.axis("off")
    plt.show()
