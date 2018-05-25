import unittest
import cv2

from gray_level_profile import GrayLevelProfile


class GrayLevelProfileTest(unittest.TestCase):
    def test_image_translation(self):
        im = cv2.imread(
            "/Users/joao/Projects/computer_vision_class/xray_teeth_detection/data/Radiographs/01.tif",
            cv2.IMREAD_GRAYSCALE)
        cv2.imshow("testtt", im)
        _, res_img = GrayLevelProfile.from_image(im, (1522, 955), (1, 1), 2)
        cv2.imwrite("./glp_test.png", res_img)