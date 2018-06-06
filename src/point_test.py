import unittest

from point import *


class PointTest(unittest.TestCase):
    def test_add_noise(self):
        p = Point(1, 2)
        new_p = Point.add_noise(p)
        print(p)
