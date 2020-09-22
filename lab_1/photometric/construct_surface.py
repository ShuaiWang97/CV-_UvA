import numpy as np

def construct_surface(p, q, path_type='row'):

    '''
    CONSTRUCT_SURFACE construct the surface function represented as height_map
       p : measures value of df / dx
       q : measures value of df / dy
       path_type: type of path to construct height_map, either 'column',
       'row', or 'average'
       height_map: the reconstructed surface
    '''
    
    h, w = p.shape
    height_map = np.zeros([h, w])
    
    if path_type=='column':
        for i in range(1, h):
        	height_map[i, 0] = height_map[i-1 , 0] + q[i-1, 0];

        for j in range(h):
        	for k in range(1, w):
        		height_map[j, k] = height_map[j, k-1] + p[j, k-1]
            
    elif path_type=='row':
        height_map = np.transpose(construct_surface(np.transpose(q), np.transpose(p), 'column'))
        
    elif path_type=='average':
        height_map = np.mean([construct_surface(p, q, 'column'), construct_surface(p, q, 'row')], axis=0)
        
    return height_map
        
