import numpy as np
import time

def estimate_alb_nrm( image_stack, scriptV, shadow_trick=True):
    
    # COMPUTE_SURFACE_GRADIENT compute the gradient of the surface
    # INPUT:
    # image_stack : the images of the desired surface stacked up on the 3rd dimension
    # scriptV : matrix V (in the algorithm) of source and camera information
    # shadow_trick: (true/false) whether or not to use shadow trick in solving linear equations
    # OUTPUT:
    # albedo : the surface albedo
    # normal : the surface normal
    
    preferMatrixCalc = False
    
    h, w, n = image_stack.shape
    
    if preferMatrixCalc:
      i = np.array([[z for z in image_stack[y, x]] for y in range(h) for x in range(w)]).reshape(h, w, n)
      scriptI = np.apply_along_axis(np.diag, 2, i)

      l = np.einsum('ijkl,ijk->ijk',scriptI, i) if shadow_trick else i
      r = np.matmul(scriptI, scriptV) if shadow_trick else scriptV
      
      g = np.array([np.linalg.lstsq(r[y, x] if shadow_trick else r, l[y, x], rcond=None)[0] for y in range(h) for x in range(w)]).reshape(h, w, 3)
      
      albedo = np.linalg.norm(g, axis=2)
      
      with np.errstate(divide='ignore', invalid='ignore'):
        normal = np.array([np.divide(g[y,x], albedo[y,x]) for y in range(h) for x in range(w)]).reshape(h, w, 3)
        
    else:
      # Shadow trick is not implemented as this code will only run for large amounts of pictures
      # For which we don't wish to use the trick
      
      albedo = np.zeros([h, w])
      normal = np.zeros([h, w, 3])  
      g = np.zeros([h, w, 3])
    
      for ix in range(image_stack.shape[0]):
        for iy in range(image_stack.shape[1]):
          g[ix,iy] = np.dot(np.dot(np.linalg.pinv(np.dot(scriptV.T,scriptV)),scriptV.T),image_stack[ix,iy])
          albedo[ix,iy] = np.sqrt(g[ix,iy,0]**2 + g[ix,iy,1]**2 + g[ix,iy,2]**2)
          with np.errstate(divide='ignore', invalid='ignore'):
            normal[ix,iy] = g[ix,iy] / albedo[ix,iy]
    
    return albedo, normal


if __name__ == '__main__':
    n = 5
    image_stack = np.zeros([10,10,n])
    scriptV = np.zeros([n,3])
    estimate_alb_nrm( image_stack, scriptV, shadow_trick=True)