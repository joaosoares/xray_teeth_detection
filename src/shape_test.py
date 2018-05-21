import unittest

import numpy.testing as npt

from shape import Shape


class TestShape(unittest.TestCase):
    def test_translate_all_to_origin(self):
        shapes = [
            Shape([(1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (3, 3), (3, 2),
                   (2, 2)]),
            Shape([(11, 2), (11, 3), (11, 4), (12, 4), (13, 4), (13, 3),
                   (13, 2), (12, 2)]),
        ]

        Shape.translate_all_to_origin(shapes)
        self.assertEqual(shapes[0].points.all(), shapes[1].points.all())

    def test_align_shapes(self):
        s1 = Shape([(1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (3, 3), (3, 2),
                    (2, 2)])
        s2 = Shape([(5, 2), (6, 3), (7, 4), (8, 3), (9, 2), (8, 1), (7, 0),
                    (6, 1)])

        Shape.translate_all_to_origin([s1, s2])
        s2.align(s1)

        npt.assert_almost_equal(s1.points, s2.points, decimal=1)


if __name__ == '__main__':
    unittest.main()
