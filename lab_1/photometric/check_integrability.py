import numpy as np

def check_integrability(normals):
    #  CHECK_INTEGRABILITY check the surface gradient is acceptable
    #   normals: normal image
    #   p : df / dx
    #   q : df / dy
    #   SE : Squared Errors of the 2 second derivatives

    # initalization
    p = np.zeros(normals.shape[:2])
    q = np.zeros(normals.shape[:2])
    SE = np.zeros(normals.shape[:2])
    
    
    p = np.divide(normals[:,:,0], normals[:,:,2])
    q = np.divide(normals[:,:,1], normals[:,:,2])

    p[p!=p] = 0
    q[q!=q] = 0
     
    p2 = np.gradient(p, axis = 1) 
    q2 = np.gradient(q, axis = 0)
    
    
    SE= (np.array(p2)-np.array(q2))**2
    return p, q, SE


if __name__ == '__main__':
    normals = np.zeros([10,10,3])
    check_integrability(normals)