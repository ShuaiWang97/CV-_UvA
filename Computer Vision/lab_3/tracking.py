from harris import harris_corner_detector
from lucas_kanade import lucas_kanade

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
# from PIL import image


def tracking(images):

        # fig, ax = plt.subplots()
        fig = plt.gcf()
        fig.show()
        fig.canvas.draw()

        for i in range(len(images)):
            j = i+1
            image1 = images[i]
            image2 = images[j]

            H, r, c = harris_corner_detector(image1, plot=False, sobel=True)
            X, Y, U, V = lucas_kanade(image1, image2, onlyMotions = False)

            X = np.array(X.T)
            Y = np.array(Y.T)

            #Make some empty lists to fill
            x = []
            y = []
            u = []
            v = []

            # Loop over all points found with the harris algorithm
            for k in range(len(r)):
                rx = int(r[k])
                cy = int(c[k])

                x.append(X[rx][cy])
                y.append(Y[rx][cy])
                u.append(U[rx][cy])
                v.append(V[rx][cy])

            #Make them np arrays again
            X = np.array(x)
            Y = np.array(y)
            U = np.array(u)
            V = np.array(v)

            #Plot everything in one figure
            plt.clf()
            plt.imshow(image1, cmap="gray")
            plt.quiver(Y, X, U, V)
            plt.scatter(c, r, marker='x')

            plt.savefig("plots/%d.jpg"%(i))

            fig.canvas.draw()

def demo():

    images = [cv2.imread(file, 0) for file in sorted(glob.glob("person_toy/*.jpg"))]
    # images = [cv2.imread(file, 0) for file in sorted(glob.glob("pingpong/*.jpeg"))]
    tracking(images)

# https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/
# https://www.geeksforgeeks.org/python-create-video-using-multiple-images-using-opencv/

demo()
