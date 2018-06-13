"""Assorted utilities for tests"""
import unittest
from typing import Tuple, List

import cv2
import numpy as np
import numpy.testing as npt

from active_shape_model import ActiveShapeModel
from image_shape import ImageShape
from imgutils import load_images, apply_median_blur, apply_sobel
from incisors import Incisors
from shape import Shape


def create_diagonal_test_image(l: int) -> np.ndarray:
    diagonal = np.linspace(0, 255, l // 2, dtype=np.uint8)
    if l % 2 == 0:
        diagonal = np.append(diagonal, np.flipud(diagonal))
    else:
        diagonal = np.append(diagonal, [np.array(255), np.flipud(diagonal)])

    return np.invert(np.diag(diagonal), dtype=np.uint8)


def create_circle_at_origin(l: int = 50, r: int = 20, m: int = 40) -> np.ndarray:
    if 2 * r > l:
        raise ValueError("Diameter of circle bigger than image dimensions")
    im = np.zeros((l, l), dtype=np.uint8)
    rows, cols = im.shape
    cv2.circle(im, (rows // 2, cols // 2), r, (255, 255, 255))
    points = np.transpose(np.where(im == 255))
    n = points.shape[0]
    shape_point_indexes = [i * n // m + n // (2 * m) for i in range(m)]
    shape_points = points[shape_point_indexes]
    return ImageShape(im, Shape(shape_points))


TEST_IMAGES = {
    "center_diagonal_line": np.array(
        [
            [255, 255, 255, 255, 0],
            [255, 255, 255, 0, 255],
            [255, 255, 0, 255, 255],
            [255, 0, 255, 255, 255],
            [0, 255, 255, 255, 255],
        ],
        np.uint8,
    ),
    "center_horizontal_line": np.array(
        [
            [255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255],
            [0, 0, 0, 0, 0],
            [255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255],
        ],
        np.uint8,
    ),
    "center_diagonal_line_long": create_diagonal_test_image(50),
    "center_circle_long": create_circle_at_origin(50, 15),
}


def load_incisor(
    incisor=Incisors.UPPER_OUTER_LEFT, extra_text="", blur=False, sobel=False
) -> Tuple[ActiveShapeModel, List[ImageShape]]:
    """Loads asm and imgshapes for sample incisor"""
    images = load_images(range(1, 15))
    blurred_images = apply_median_blur(images, times=3)
    if blur:
        images = blurred_images
    sobel_images = apply_sobel(images)
    if sobel:
        images = sobel_images
    asms, imgshapes = Incisors.active_shape_models(images, [incisor])
    asm: ActiveShapeModel = asms[incisor]
    imgshape: List[ImageShape] = imgshapes[incisor]
    return asm, imgshape


def image_shape_with_noise(image_shape: ImageShape) -> ImageShape:
    points = image_shape.shape.as_point_list()
    points_with_noise = [point.add_noise() for point in points]
    shape_with_noise = Shape(points_with_noise)
    return ImageShape(image_shape.image, shape_with_noise)


class ShapeAssertions:
    def assert_shape_equal(self, actual: Shape, desired: Shape):
        npt.assert_equal(actual.points, desired.points)

    def assert_shape_approx_equal(self, actual: Shape, desired: Shape, significant=7):
        npt.assert_approx_equal(actual.points, desired.points, significant)
