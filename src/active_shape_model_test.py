import unittest

import numpy as np
import numpy.testing as npt
from sklearn.decomposition import PCA

from active_shape_model import ActiveShapeModel
from shape import Shape
from image_shape import ImageShape
from point import Point
from incisors import Incisors
from testutils import image_shape_with_noise, load_incisor
from shapeutils import plot_shape, plot_image_shape


class ActiveShapeModelTest(unittest.TestCase):
    def test_from_image_shapes(self):
        shapes = [
            Shape([(1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (3, 3), (3, 2), (2, 2)]),
            Shape([(5, 2), (6, 3), (7, 4), (8, 3), (9, 2), (8, 1), (7, 0), (6, 1)]),
            Shape([(5, 2), (6, 3), (8, 4), (8, 3), (9, 2), (8, 1), (7, 0), (6, 1)]),
            Shape([(5, 2), (6, 3), (6, 4), (8, 3), (9, 2), (8, 1), (7, 0), (6, 1)]),
            Shape([(5, 2), (6, 2), (8, 4), (8, 3), (9, 2), (8, 1), (7, 0), (6, 1)]),
            Shape([(5, 2), (6, 4), (8, 4), (8, 3), (9, 2), (8, 1), (7, 0), (6, 1)]),
        ]

        am = ActiveShapeModel.from_image_shapes(shapes)

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

    def test_match_target(self):
        pass

    def test_propose_shape(self):
        # Arrange
        asm, image_shapes = load_incisor(blur=True, sobel=True)
        # Manually fit
        bottom_left = Point(1310, 745)
        top_right = Point(1410, 1000)
        original_imgshp = ImageShape(
            image_shapes[0].image,
            asm.mean_shape.conform_to_rect(bottom_left, top_right),
        )

        # Act
        imageshape = original_imgshp
        for i in range(5):
            proposed_shape = asm.propose_shape(imageshape)
            matched_shape, *_ = asm.match_target(proposed_shape)
            # Assert
            plot_image_shape(imageshape, display=False)
            plot_shape([proposed_shape, matched_shape])
            imageshape = ImageShape(imageshape.image, matched_shape)

    def test_fit_to_image(self):
        asm, image_shapes = load_incisor(
            blur=False, sobel=False, incisor=Incisors.LOWER_INNER_LEFT
        )
        # Manually fit
        bottom_left = Point(1431, 992)
        top_right = Point(1544, 1274)
        initial_imgshp = ImageShape(
            image_shapes[4].image,
            asm.mean_shape.conform_to_rect(bottom_left, top_right),
        )

        matched_image_shape = asm.fit_to_image(initial_imgshp)

        plot_image_shape(
            initial_imgshp, display=False, interpol=False, dots=False, vecs=False
        )
        plot_image_shape(matched_image_shape)


if __name__ == "__main__":
    unittest.main()
