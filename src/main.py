import logging
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np

from incisors import Incisors


def load_images(indices):
    """
    Returns a list with opened cv2 images in the data folder
    corresponding to the indices that are passed in.
    """
    image_filenames = ["./data/Radiographs/{:02d}.tif".format(i) for i in indices]
    images = [
        cv2.imread(filename, flags=cv2.IMREAD_GRAYSCALE) for filename in image_filenames
    ]
    return images


def main():
    images = load_images(range(1, 15))
    active_shape_models, image_shapes = Incisors.active_shape_models(
        images, incisors=[Incisors.UPPER_OUTER_LEFT]
    )

    asm = active_shape_models[Incisors.UPPER_OUTER_LEFT]
    imgshp = image_shapes[Incisors.UPPER_OUTER_LEFT]

    shap = asm.propose_shape(imgshp[0])

    return


if __name__ == "__main__":
    main()
