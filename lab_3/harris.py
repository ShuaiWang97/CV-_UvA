from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import cv2
import scipy
from scipy.signal import convolve2d
import math
from scipy import ndimage

def harris_corner_detector(im, n = 5, f = 0.1, plot = False, sobel = False):

  # computing cornerness value for every pixel in image
  if sobel:
    imDx = convolve2d(im, np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]), mode='same', boundary='symm')
    imDy = convolve2d(im, np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]), mode='same', boundary='symm')
  else:
    imDx = convolve2d(im, np.array([[1, 0, -1],] * 3), mode='same', boundary='symm')
    imDy = convolve2d(im, np.array([[1, 0, -1],] * 3).T, mode='same', boundary='symm')

  A = gaussian_filter(imDx ** 2, sigma=1)
  B = gaussian_filter(imDx * imDy, sigma=1)
  C = gaussian_filter(imDy ** 2, sigma=1)

  H = (A * C - B ** 2) - 0.04 * (A + C) ** 2

  if plot:
    plt.imshow(imDx, cmap='gray')
    plt.show()
    plt.imshow(imDy, cmap='gray')
    plt.show()

  # parsing cornerness values using window of size nxn

  t = f*np.max(H) # defining threshold as a factor of max value in H
  h, w = H.shape # height and width of input image
  d = math.floor(n/2) # length of sides of square window in pixels
  m = (n**2-1)/2 # center index of window (assuming 0-indexing as in Python ofc)

  r = []
  c = []

  for i in range(d, h - d):
    for j in range(d, w - d):
      # for every window

      W = H[i-d:i+d+1, j-d:j+d+1]

      if np.argmax(W) == m and W[d, d] > t:
        r.append(i)
        c.append(j)

  if plot:
    plt.scatter(c, r, marker='x')
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.show()

  return (H, r, c)



def demo():
  im1 = cv2.imread("person_toy/00000001.jpg", 0)
  #im145 = rotated = ndimage.rotate(im1, 45)
  im2 = cv2.imread("pingpong/0000.jpeg", 0)

    """
  code used to create figure 1:

  fvals = np.linspace(0.0005, 0.9, 20)
  numbers1 = []
  numbers2 = []

  for fval in fvals:
    (h1, r1, c2) = harris_corner_detector(im1, f=fval)
    numbers1.append(len(r1))
    (h2, r2, c2) = harris_corner_detector(im2, f=fval)
    numbers2.append(len(r2))

  plt.plot(fvals, numbers1, label='00000001.jpg')
  plt.plot(fvals, numbers2, label='0000.jpeg')
  plt.legend()
  plt.xlabel('f')
  plt.ylabel('number of corners found')
  plt.show()

  """
  harris_corner_detector(im1, plot=True)
  #harris_corner_detector(im2, plot=True)


# demo()
