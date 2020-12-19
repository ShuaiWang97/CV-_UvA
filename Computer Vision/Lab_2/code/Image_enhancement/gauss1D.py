import numpy as np


def gauss1D( sigma , kernel_size):
    G = np.zeros((1, kernel_size))
    if kernel_size % 2 == 0:
        raise ValueError('kernel_size must be odd, otherwise the filter will not have a center to convolve on')
    # solution
    x_values = np.array([range((-kernel_size+1) // 2, (kernel_size+1)//2)])


    #calculate G
    G= np.exp(-x_values**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    G=(1/np.sum(G))*G

    return G

 

if __name__ == '__main__':
    sigma = 2
    kernel_size = 5
    gauss1D( sigma, kernel_size)