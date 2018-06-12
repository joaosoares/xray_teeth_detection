import unittest

import cv2
import numpy as np
import numpy.testing as npt

from gray_level_profile import GrayLevelProfile
from point import Point
from shapeutils import plot_point, plot_vecs
from testutils import TEST_IMAGES


class GrayLevelProfileTest(unittest.TestCase):
    def test_rotate_and_center_image(self):
        im = TEST_IMAGES["center_diagonal_line"]
        res_img, _, _ = GrayLevelProfile.rotate_and_center_image(im, (2, 2), (1, 1))
        # cv2.imwrite("./glp_test.png", res_img)

    def test_all_from_image_shapes(self):
        cv2.imshow("test", TEST_IMAGES["center_circle_long"])
        cv2.waitKey()
        image1 = np.zeros((5, 5), np.uint8)

    def test_sliding_profiles(self):
        im = TEST_IMAGES["center_diagonal_line_long"]
        profiles = GrayLevelProfile.sliding_profiles(im, (25, 25), (1, -1), 4)
        self.assertEqual(len(profiles), 9, "Wrong number of profiles generated")
        self.assertTupleEqual(profiles[0].shape, (9,))

    def test_get_point_position(self):
        """Test for GrayLevelProfile#get_point_position"""
        prof_idx = 3
        cnt = 20  # to guarantee that idx 3 is to the right
        cnt_pnt = Point(10, 10)
        dir_vec = Point(3, 3)
        p = GrayLevelProfile.get_point_position(prof_idx, cnt, cnt_pnt, dir_vec)

        print(p)
        plot_point(cnt_pnt, display=False)
        plot_vecs(dir_vec, cnt_pnt, display=False)
        plot_point(p)
        npt.assert_equal(p.x, 12)
        npt.assert_equal(p.y, 12)


if __name__ == "__main__":
    unittest.main()
