import unittest

import numpy as np
import numpy.testing as npt
from sklearn.decomposition import PCA

from active_shape_model import ActiveShapeModel
from shape import Shape


class ActiveShapeModelTest(unittest.TestCase):
    def test_from_shapes(self):
        shapes = [
            Shape([(1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (3, 3), (3, 2), (2, 2)]),
            Shape([(5, 2), (6, 3), (7, 4), (8, 3), (9, 2), (8, 1), (7, 0), (6, 1)]),
            Shape([(5, 2), (6, 3), (8, 4), (8, 3), (9, 2), (8, 1), (7, 0), (6, 1)]),
            Shape([(5, 2), (6, 3), (6, 4), (8, 3), (9, 2), (8, 1), (7, 0), (6, 1)]),
            Shape([(5, 2), (6, 2), (8, 4), (8, 3), (9, 2), (8, 1), (7, 0), (6, 1)]),
            Shape([(5, 2), (6, 4), (8, 4), (8, 3), (9, 2), (8, 1), (7, 0), (6, 1)]),
        ]

        am = ActiveShapeModel.from_shapes(shapes)

        pca_len = len(am)

        standard_pca = PCA(n_components=pca_len)
        data = standard_pca.fit(np.array([shape.as_vector() for shape in shapes]))
        eigenvectors = standard_pca.components_
        eingenvalues = standard_pca.explained_variance_

        npt.assert_almost_equal(am.eigenvectors, eigenvectors)

    def test_align_shapes(self):
        s1 = Shape([(1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (3, 3), (3, 2), (2, 2)])
        s2 = Shape([(5, 2), (6, 3), (7, 4), (8, 3), (9, 2), (8, 1), (7, 0), (6, 1)])

        Shape.translate_all_to_origin([s1, s2])
        s2.align(s1)

        npt.assert_almost_equal(s1.points, s2.points, decimal=1)


if __name__ == "__main__":
    unittest.main()
