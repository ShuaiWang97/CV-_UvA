from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import cv2
import scipy
from scipy.signal import convolve2d
import math
from scipy import ndimage
import sys
import glob

def lucas_kanade(im1, im2, n = 15, str = 10, onlyMotions = False):
  kDt = np.array([[1/9] * 3,] * 3)

  imDx = convolve2d(im1, np.array([[1, 0, -1],] * 3), mode='same', boundary='symm')
  imDy = convolve2d(im1, np.array([[1, 0, -1],] * 3).T, mode='same', boundary='symm')
  imDt1 = convolve2d(im2, kDt, mode='same', boundary='symm')
  imDt2 = convolve2d(im1, -kDt, mode='same', boundary='symm')
  imDt  = imDt1 + imDt2

  h, w = im1.shape # height and width of input images
  d = math.floor(n/2) # length of sides of square window in pixels

  if onlyMotions:
    X, Y = np.array(np.meshgrid(np.arange(d, h - d), np.arange(d, w - d)))
    U = np.zeros((h - 2*d, w - 2*d))
    V = np.zeros((h - 2*d, w - 2*d))
  else:
    X, Y = np.array(np.meshgrid(np.arange(0, h), np.arange(0, w)))
    U = np.zeros((h, w))
    V = np.zeros((h, w))

  for i in range(d, h - d):
    for j in range(d, w - d):
      # for every window

      A = np.array([imDx[i-d:i+d+1, j-d:j+d+1].flatten(), imDy[i-d:i+d+1, j-d:j+d+1].flatten()]).T
      b = np.array(imDt[i-d:i+d+1, j-d:j+d+1].flatten()).T

      ATA = np.matmul(A.T, A)

      if np.linalg.cond(ATA) < 1/sys.float_info.epsilon: # checking wheter A is not singular
        v = np.matmul(np.matmul(np.linalg.inv(ATA), A.T), b)
      else:
        v = [0., 0.]

      if onlyMotions:
        U[i-d, j-d] = v[0]
        V[i-d, j-d] = v[1]
      else:
        U[i, j] = v[0]
        V[i, j] = v[1]

  if onlyMotions:
    c = 'b'
  else:
    # plt.imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
    c = 'w'

  # plt.quiver(X[::str, ::str], Y[::str, ::str], U[::str, ::str], V[::str, ::str], color=c, pivot='mid', angles='xy')
  # plt.show()

  return X, Y, U, V

def demo():
  # im1 = cv2.imread("sphere1.ppm", 0)
  # im2 = cv2.imread("sphere2.ppm", 0)
  # im21 = cv2.imread("synth1.pgm", 0)
  # im22 = cv2.imread("synth2.pgm", 0)
  #
  # # running for sphere
  # plt.imshow(im1, cmap='gray')
  # plt.show()
  #
  # lucas_kanade(im1, im2)
  #
  # # running for synth
  # plt.imshow(im21, cmap='gray')
  # plt.show()
  # lucas_kanade(im21, im22)
    images = [cv2.imread(file, 0) for file in glob.glob("pingpong/*.jpeg")]

    lucas_kanade(images[0], images[1])
    lucas_kanade(images[1], images[2])
    lucas_kanade(images[2], images[3])


# demo()
