import matplotlib.pyplot as plt
import numpy as np
import torchvision

def show_generated_images(images):
    # Create a grid of generated images and display it
    grid = np.transpose(torchvision.utils.make_grid(images, normalize=True), (1, 2, 0))
    plt.imshow(grid)
    plt.show()