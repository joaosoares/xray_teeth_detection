import unittest

import numpy as np
import numpy.testing as npt

from point import Point
from shape import Shape
from shapeutils import plot_shape, plot_vecs


class ShapeTest(unittest.TestCase):
    def test_translate_all_to_origin(self):
        shapes = [
            Shape([(1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (3, 3), (3, 2), (2, 2)]),
            Shape(
                [(11, 2), (11, 3), (11, 4), (12, 4), (13, 4), (13, 3), (13, 2), (12, 2)]
            ),
        ]

        Shape.translate_all_to_origin(shapes)
        self.assertEqual(shapes[0].points.all(), shapes[1].points.all())

    def test_align_shapes(self):
        s1 = Shape([(1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (3, 3), (3, 2), (2, 2)])
        s2 = Shape([(5, 2), (6, 3), (7, 4), (8, 3), (9, 2), (8, 1), (7, 0), (6, 1)])

        Shape.translate_all_to_origin([s1, s2])
        s2.align(s1)

        npt.assert_almost_equal(s1.points, s2.points, decimal=1)

    def test_get_orthogonal_vectors(self):
        round_shape = Shape(
            [
                (4.5, 1),
                (4, 1),
                (3.5, 1),
                (3, 1),
                (2.5, 1.5),
                (2, 2),
                (1.5, 2.5),
                (1, 3),
                (1, 3.5),
                (1, 4),
                (1.5, 4.5),
                (2, 5),
                (2.5, 5.5),
                (3, 6),
                (3.5, 6),
                (4, 6),
                (4.5, 6),
                (5, 6),
                (5.5, 5.5),
                (6, 5),
                (6.5, 4.5),
                (7, 4),
                (7, 3.5),
                (7, 3),
                (6.5, 2.5),
                (6, 2),
                (5.5, 1.5),
                (5, 1),
            ]
        )
        ort_vects = round_shape.get_orthogonal_vectors()
        print(ort_vects)

        plot_shape(round_shape, plot=False)
        plot_vecs(ort_vects, round_shape.as_point_list())

    def test_as_point_list(self):
        s1 = Shape([(1, 2), (2, 1)])
        self.assertListEqual(s1.as_point_list(), [Point(1, 2), Point(2, 1)])

    def test_x_vector(self):
        s1 = Shape([(1, 2), (3, 4), (5, 6)])
        x_vec = s1.x_vector()
        npt.assert_array_equal(x_vec, np.array([1, 3, 5]))

    def test_y_vector(self):
        s1 = Shape([(1, 2), (3, 4), (5, 6)])
        y_vec = s1.y_vector()
        npt.assert_array_equal(y_vec, np.array([2, 4, 6]))


if __name__ == "__main__":
    unittest.main()
