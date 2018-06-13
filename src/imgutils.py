"""Set of utilities to handle loading and saving images"""

import cv2


def load_images(indices, extra_text="", flags=cv2.IMREAD_GRAYSCALE):
    """
    Returns a list with opened cv2 images in the data folder
    corresponding to the indices that are passed in.
    """
    image_filenames = [
        "./data/Radiographs/{:02d}{}.tif".format(i, extra_text) for i in indices
    ]
    images = [cv2.imread(filename, flags=flags) for filename in image_filenames]
    return images


def apply_median_blur(images, kernel_size=5, times=1):
    """Applies median blur to a set of images"""
    for i in range(times):
        images = [cv2.medianBlur(image, ksize=kernel_size) for image in images]
    return images


def top_hat_processing(images, ksize=150):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(ksize, ksize))
    return [cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel=kernel) for img in images]
