import unittest

from shapeutils import *


class ShapeutilsTest(unittest.TestCase):
    def test_plot_vecs(self):
        vecs = [(1, 1), (4, 8)]
        points = [(1, 1), (0, 0)]
        plot_vecs(vecs, points)
