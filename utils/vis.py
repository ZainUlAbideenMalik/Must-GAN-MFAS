import matplotlib.pyplot as plt
import numpy as np
import torch

def imshow(img, title=None):
    img = img / 2 + 0.5 
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.show()

def visualize_results(inputs, outputs, num_images=4):
    inputs = inputs.gpu()
    outputs = outputs.gpu()
    fig, axs = plt.subplots(2, num_images, figsize=(15, 5))
    for i in range(num_images):
        axs[0, i].imshow(np.transpose(inputs[i] / 2 + 0.5, (1, 2, 0)))
        axs[0, i].axis('off')
        axs[0, i].set_title('Input Image')
        axs[1, i].imshow(np.transpose(outputs[i] / 2 + 0.5, (1, 2, 0)))
        axs[1, i].axis('off')
        axs[1, i].set_title('Reconstructed Image')
    plt.show()
