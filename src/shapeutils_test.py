"""Tests for the shapeutils module"""
import unittest

from shapeutils import (
    plot_vecs,
    plot_shape,
    plot_interpol,
    plot_image_shape,
    plot_profile,
)
from testutils import load_incisor

from gray_level_profile import GrayLevelProfile


class ShapeutilsTest(unittest.TestCase):
    def test_plot_vecs(self):
        vecs = [(1, 1), (4, 8)]
        points = [(1, 1), (0, 0)]
        plot_vecs(vecs, points)

    def test_interpol(self):
        asm, imgshps = load_incisor()
        shape = imgshps[0].shape
        image = imgshps[0].image
        vecs, tck = shape.get_orthogonal_vectors(with_tck=True)
        plot_shape(shape, display=False, overlay_image=image)
        plot_vecs(vecs, shape.as_point_list(), display=False)
        plot_interpol(tck)

    def test_plot_image_shape(self):
        _, imgshps = load_incisor()
        plot_image_shape(imgshps[0])

    def test_plot_profile(self):
        print("Hello")
        asm, imgshps = load_incisor()
        imgshp = imgshps[0]
        image = imgshp.image
        point = imgshp.shape.as_point_list()[0]
        dir_vec = imgshp.shape.get_orthogonal_vectors()[0]
        hsz = 20
        profiles = GrayLevelProfile.sliding_profiles(image, point, dir_vec, hsz)
        plot_profile(profiles)


if __name__ == "__main__":
    unittest.main()
