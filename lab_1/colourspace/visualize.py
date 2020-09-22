import matplotlib.pyplot as plt
import numpy as np

def visualize(input_image, colourspace):

    if colourspace.lower() == 'gray':
        plt.imshow(input_image, cmap='gray', vmin=0, vmax=255)
        plt.axis("off")
    elif colourspace.lower() == 'opponent':
        plt.subplot(2,2,1)
        plt.imshow(input_image,cmap="gray")
        plt.axis("off")
        plt.title('Full image')
        plt.subplot(2,2,2)
        plt.imshow(input_image[:,:,0],cmap="gray")
        plt.axis("off")
        plt.title('O1 intensity in grayscale')
        plt.subplot(2,2,3)
        plt.imshow(input_image[:,:,1],cmap="gray")
        plt.axis("off")
        plt.title('O2 intensity in grayscale')
        plt.subplot(2,2,4)
        plt.imshow(input_image[:,:,2],cmap="gray")
        plt.axis("off")
        plt.title('O3 intensity in grayscale')

    elif colourspace.lower() == 'hsv':
        plt.subplot(2,2,1)
        plt.imshow(input_image,cmap="gray")
        plt.axis("off")
        plt.title('Full image')
        plt.subplot(2,2,2)
        plt.imshow(input_image[:,:,0],cmap="gray")
        plt.axis("off")
        plt.title('H intensity in grayscale')
        plt.subplot(2,2,3)
        plt.imshow(input_image[:,:,1],cmap="gray")
        plt.axis("off")
        plt.title('S intensity in grayscale')
        plt.subplot(2,2,4)
        plt.imshow(input_image[:,:,2],cmap="gray")
        plt.axis("off")
        plt.title('V intensity in grayscale')

    elif colourspace.lower() == 'ycbcr':
        plt.subplot(2,2,1)
        plt.imshow(input_image,cmap="gray")
        plt.axis("off")
        plt.title('Full image')
        plt.subplot(2,2,2)
        plt.imshow(input_image[:,:,0],cmap="gray")
        plt.axis("off")
        plt.title('Y intensity in grayscale')
        plt.subplot(2,2,3)
        plt.imshow(input_image[:,:,1],cmap="gray")
        plt.axis("off")
        plt.title('cb intensity in grayscale')
        plt.subplot(2,2,4)
        plt.imshow(input_image[:,:,2],cmap="gray")
        plt.axis("off")
        plt.title('cr intensity in grayscale')

    elif colourspace.lower() == 'rgb':
        plt.subplot(2,2,1)
        plt.imshow(input_image)
        plt.axis("off")
        plt.title('Full image')
        plt.subplot(2,2,2)
        plt.imshow(input_image[:,:,0])
        plt.axis("off")
        plt.title('r')
        plt.subplot(2,2,3)
        plt.imshow(input_image[:,:,1])
        plt.axis("off")
        plt.title('g')
        plt.subplot(2,2,4)
        plt.imshow(input_image[:,:,2])
        plt.axis("off")
        plt.title('b')

    else:
        plt.imshow(input_image)

    plt.show()
