"""Set of utilities to handle loading and saving images"""

import cv2


def load_images(indices, flags=cv2.IMREAD_GRAYSCALE):
    """
    Returns a list with opened cv2 images in the data folder
    corresponding to the indices that are passed in.
    """
    image_filenames = ["./data/Radiographs/{:02d}.tif".format(i) for i in indices]
    images = [cv2.imread(filename, flags=flags) for filename in image_filenames]
    return images
