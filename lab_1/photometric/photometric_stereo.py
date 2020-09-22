import numpy as np
import cv2
import os
from utils import *
from estimate_alb_nrm import estimate_alb_nrm
from check_integrability import check_integrability
from construct_surface import construct_surface

print('Part 1: Photometric Stereo\n')

def photometric_stereo(image_dir='./SphereGray5/', colour = False):

    # obtain many images in a fixed view under different illumination
    print('Loading images...\n')
    [image_stack, scriptV] = load_syn_images(image_dir)
    
    if colour:
      [image_stack1, scriptV1] = load_syn_images(image_dir, 1)
      [image_stack2, scriptV2] = load_syn_images(image_dir, 2)
      image_stack = np.concatenate((image_stack, image_stack1, image_stack2), axis=2)
      scriptV = np.concatenate((scriptV, scriptV1, scriptV2))
    
    [h, w, n] = image_stack.shape
    
    print('Finish loading %d images.\n' % n)

    # compute the surface gradient from the stack of imgs and light source mat
    print('Computing surface albedo and normal map...\n')
    [albedo, normals] = estimate_alb_nrm(image_stack, scriptV)

    # integrability check: is (dp / dy  -  dq / dx) ^ 2 small everywhere?
    print('Integrability checking\n')
    [p, q, SE] = check_integrability(normals)

    threshold = 0.005;
    print('Number of outliers: %d\n' % np.sum(SE > threshold))
    SE[SE <= threshold] = float('nan') # for good visualization

    # compute the surface height
    height_map = construct_surface( p, q )

    # show results
    show_results(albedo, normals, height_map, SE)
    
def photometric_stereo_colour(image_dir='./SphereColour/' ):

    # obtain many images in a fixed view under different illumination
    
    cValues = []
    
    for c in range(3):
        print('Loading images in channel %d...\n' % c)
        [image_stack, scriptV] = load_syn_images(image_dir, c)
        [h, w, n] = image_stack.shape
    
        print('Finish loading %d images.\n' % n)

        # compute the surface gradient from the stack of imgs and light source mat
        print('Computing surface albedo and normal map...\n')
        [albedo, normals, i] = estimate_alb_nrm(image_stack, scriptV, True, True)
        
        print(estimate_k_value(normals, i))
        
        quit()

        # integrability check: is (dp / dy  -  dq / dx) ^ 2 small everywhere?
        print('Integrability checking\n')
        cValues.append(check_integrability(normals))

        threshold = 0.005;
        print('Number of outliers: %d\n' % np.sum(SE > threshold))
        SE[SE <= threshold] = float('nan') # for good visualization

    # compute the surface height
    height_map = construct_surface( p, q )

    # show results
    show_results(albedo, normals, height_map, SE)

## Face
def photometric_stereo_face(image_dir='./yaleB02/'):
    [image_stack, scriptV] = load_face_images(image_dir)
    [h, w, n] = image_stack.shape
    print('Finish loading %d images.\n' % n)
    print('Computing surface albedo and normal map...\n')
    albedo, normals = estimate_alb_nrm(image_stack, scriptV)

    # integrability check: is (dp / dy  -  dq / dx) ^ 2 small everywhere?
    print('Integrability checking')
    p, q, SE = check_integrability(normals)

    threshold = 0.005;
    print('Number of outliers: %d\n' % np.sum(SE > threshold))
    SE[SE <= threshold] = float('nan') # for good visualization

    # compute the surface height
    height_map = construct_surface( p, q )

    # show results
    show_results(albedo, normals, height_map, SE)
    
if __name__ == '__main__':
    # To run with colours:
    
    #photometric_stereo('./SphereColor/', True)
    
    
    # To run with a sole channel:
    
    
    photometric_stereo('./SphereGray25/')
    
    
    #photometric_stereo_face()