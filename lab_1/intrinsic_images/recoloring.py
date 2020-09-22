import numpy as np
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    original_path = 'D:/UvA_courses/Computer_Vision/Lab1/lab1/intrinsic_images/ball.png'
    albedo_path = 'D:/UvA_courses/Computer_Vision/Lab1/lab1/intrinsic_images/ball_albedo.png'
    shading_path = 'D:/UvA_courses/Computer_Vision/Lab1/lab1/intrinsic_images/ball_shading.png'
    # Read with opencv
    original = plt.imread(original_path)#.astype(np.float32)
    albedo = plt.imread(albedo_path)#.astype(np.float32)
    shading = plt.imread(shading_path)#.astype(np.float32)
    print(albedo.shape)
    print(shading)
    
    #read RGB value splitly
    R = albedo[:,:,0]
    G = albedo[:,:,1]
    B = albedo[:,:,2]
    
    
    # Choose non 0 value
    R_val = np.array(R!= 0)
    G_val = np.array(G!= 0)
    B_val = np.array(B!= 0)

    
    #recolor to green
    R[R_val] = 0
    G[G_val] = 1
    B[B_val] = 0
    
    
    #get the final image
    constructed = np.empty(original.shape)
    #constructed[:, :] = np.array([R, G, B])
    constructed[:, :, 0] = R * shading
    constructed[:, :, 1] = G * shading
    constructed[:, :, 2] = B * shading
    
    
    
    #Show results

    plt.imshow(constructed)
    plt.axis('off')

    plt.show()
    