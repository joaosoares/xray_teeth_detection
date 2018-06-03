import unittest
import cv2

import numpy as np

from gray_level_profile import GrayLevelProfile

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
    ], np.uint8)
}
# yapf: enable


class GrayLevelProfileTest(unittest.TestCase):
    def test_rotate_and_center_image(self):
        im = test_images['center_diagonal_line']
        res_img, _, _ = GrayLevelProfile.rotate_and_center_image(
            im, (2, 2), (1, 1))
        cv2.imwrite("./glp_test.png", res_img)

    def test_all_from_image_shapes(self):
        image1 = np.zeros((5, 5), np.uint8)
        image1[:, :] = 255
        image1[2, 2] = 0
        image1[2, 1] = 127
        image1[2, 3] = 127
        cv2.imshow("test", image1)
        cv2.waitKey()