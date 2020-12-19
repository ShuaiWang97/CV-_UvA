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


    shape = original.shape
    constructed = np.empty(shape)


    for x, y in np.ndindex(shape[0], shape[1]):
        Rc = shading[x,y] * albedo[x,y,0]
        Gc = shading[x,y] * albedo[x,y,1]
        Bc = shading[x,y] * albedo[x,y,2]
        constructed[x, y] = np.array([Rc, Gc, Bc])

        

    # np.uint8(constructed)

    

    plt.imshow(albedo)
    plt.axis('off')

    plt.show()
