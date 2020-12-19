import numpy as np
from gauss1D import gauss1D

def gauss2D( sigma , kernel_size ):
    ## solution
    g_x = np.zeros((1,kernel_size))
    g_y = np.zeros((1,kernel_size))
    
    #calculate g_x and g_y
    g_x = gauss1D(sigma, kernel_size)
    g_y = gauss1D(sigma, kernel_size)
    
    #G = g_x .* g_y, shape should be5*5
    G = np.dot(np.transpose(g_x) ,g_y)
    print(G)
    return G
    
    
if __name__ == '__main__':
    sigma = 2
    kernel_size = 5
    gauss2D( sigma, kernel_size)