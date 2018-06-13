"""Set of utilities to handle loading and saving images"""

from typing import List, Union

import cv2
import numpy as np


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


def apply_sobel(
    images: Union[np.ndarray, List[np.ndarray]], scale=1, delta=0
) -> List[np.ndarray]:
    if isinstance(images, np.ndarray):
        images = [images]
    return [_apply_sobel_single_img(image, scale, delta) for image in images]


def _apply_sobel_single_img(image: np.ndarray, scale, delta) -> np.ndarray:
    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(
        image,
        ddepth,
        1,
        0,
        ksize=-1,
        scale=scale,
        delta=delta,
        borderType=cv2.BORDER_DEFAULT,
    )
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(
        image,
        ddepth,
        0,
        1,
        ksize=3,
        scale=scale,
        delta=delta,
        borderType=cv2.BORDER_DEFAULT,
    )

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
