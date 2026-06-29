import matplotlib.pyplot as plt
import numpy as np


def show_image(image, title="Image", cmap=None):

    plt.figure(figsize=(6, 6))

   # plt.imshow(image, cmap=cmap)

    plt.title(title)

    plt.axis("off")

    plt.show()


def compare_images(thermal, sr, rgb):

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(thermal, cmap="gray")
    plt.title("Thermal")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(sr, cmap="gray")
    plt.title("Super Resolution")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(rgb)
    plt.title("Colorized RGB")
    plt.axis("off")

    plt.tight_layout()
    plt.show()