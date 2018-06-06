import unittest
import cv2

import numpy as np
import numpy.testing as npt

from gray_level_profile import GrayLevelProfile
from testutils import test_images


class GrayLevelProfileTest(unittest.TestCase):
    def test_rotate_and_center_image(self):
        im = test_images["center_diagonal_line"]
        res_img, _, _ = GrayLevelProfile.rotate_and_center_image(im, (2, 2), (1, 1))
        # cv2.imwrite("./glp_test.png", res_img)

    def test_all_from_image_shapes(self):
        cv2.imshow("test", test_images["center_circle_long"])
        cv2.waitKey()
        image1 = np.zeros((5, 5), np.uint8)

    def test_sliding_profiles(self):
        im = test_images["center_diagonal_line_long"]
        profiles = GrayLevelProfile.sliding_profiles(im, (25, 25), (1, -1), 4)
        self.assertEqual(len(profiles), 9, "Wrong number of profiles generated")
        self.assertTupleEqual(profiles[0].shape, (9,))


if __name__ == "__main__":
    unittest.main()
