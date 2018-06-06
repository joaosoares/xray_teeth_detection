import numpy as np
import cv2

from image_shape import ImageShape
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
    quadrants = ()
    points = np.transpose(np.where(im == 255))
    n = points.shape[0]
    shape_point_indexes = [i * n // m + n // (2 * m) for i in range(m)]
    shape_points = points[shape_point_indexes]
    return ImageShape(im, Shape(shape_points))


# yapf: disable
test_images = {
    'center_diagonal_line': np.array([
        [255, 255, 255, 255,   0],
        [255, 255, 255,   0, 255],
        [255, 255,   0, 255, 255],
        [255,   0, 255, 255, 255],
        [  0, 255, 255, 255, 255]
    ], np.uint8),
    'center_horizontal_line': np.array([
        [255, 255, 255, 255, 255],
        [255, 255, 255, 255, 255],
        [  0,   0,   0,   0,   0],
        [255, 255, 255, 255, 255],
        [255, 255, 255, 255, 255],
    ], np.uint8),
    'center_diagonal_line_long': create_diagonal_test_image(50),
    'center_circle_long': create_circle_at_origin(50, 15)
}
# yapf: enable
