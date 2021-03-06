"""Procrustes method library"""

from itertools import chain
from typing import List, NewType, Tuple, Union, Any

import cv2
import numpy as np
from scipy import interpolate

from point import Point


class Shape:
    def __init__(
        self, points: Union[List[Point], np.ndarray], gray_level_profiles=[]
    ) -> None:
        self.points = np.array(points)

    def __len__(self):
        return np.shape(self.points)[0]

    def norm(self):
        """Returns the norm of a vector containing all points"""
        return np.linalg.norm(self.points)

    def as_vector(self) -> np.ndarray:
        """Returns points of shape as single-row np array"""
        # maybe new array creation unnecessary
        return np.reshape(np.array(self.points), (-1))

    def as_point_list(self) -> List[Point]:
        """Returns points as a list of named Point tuples"""
        return [Point(*p) for p in self.points]

    def x_vector(self):
        """Returns all the x values for each point as a vector"""
        return np.reshape(self.points[:, 0], (-1))

    def y_vector(self):
        """Returns all the y values for each point as a vector"""
        return np.reshape(self.points[:, 1], (-1))

    # This function may have too much fluctuation
    def align(self, original_ref_shape, modify=True):
        """
        Aligns shape to another reference shape

        Aligns by minimizing the weighted sum described in Eq. 2 of Cootes et al.
        """

        # Create copy of ref_shape
        centered_ref_shape = Shape(np.copy(original_ref_shape.points))
        centered_ref_shape.translate_to_origin()

        # Save original axis means
        original_axis_means = self.axis_means()

        self.translate_to_origin()

        # a coefficient
        a = np.dot(
            np.reshape(self.points, (-1)), np.reshape(centered_ref_shape.points, (-1))
        ) / (self.norm() ** 2)
        # b coefficient
        n = len(self)
        b_sum = 0
        for i in range(n):
            b_sum += (
                self.points[i][0] * centered_ref_shape.points[i][1]
                - centered_ref_shape.points[i][0] * self.points[i][1]
            )
        b = b_sum / (self.norm() ** 2)
        # s and theta coeffs
        s = np.sqrt(a ** 2 + b ** 2)
        theta = np.tanh(b / a)
        # rotation matrix
        rot_mat = np.array(
            [(np.cos(theta), np.sin(theta)), (-np.sin(theta), np.cos(theta))]
        )

        # translation matrix
        translation_cell = np.array(
            [original_ref_shape.axis_means() - self.axis_means()]
        )
        transl_mat = np.repeat(translation_cell, len(self), axis=0)

        # final point calculation, only if we want to modify the current shape
        if modify:
            self.points = s * np.matmul(self.points, rot_mat) + transl_mat

        # Ammend translation matrix with original means
        translation_cell = np.array(
            [original_ref_shape.axis_means() - original_axis_means]
        )
        transl_mat = np.repeat(translation_cell, len(self), axis=0)

        return s, rot_mat, transl_mat

    def conform_to_rect(self, bottom_left: Point, top_right: Point) -> "Shape":
        """Returns a version of the shape that fits on the given rectangle points"""
        mean = (bottom_left + top_right) / 2
        width, height = top_right - bottom_left

        shape_zero_mean = self.translated_to_origin()

        # Determine maximum and minimum dimensions
        max_x, max_y = np.max(shape_zero_mean.points, axis=0)
        min_x, min_y = np.min(shape_zero_mean.points, axis=0)

        # Scaling factor is the
        max_width = np.max(np.abs([max_x, min_x]))
        max_height = np.max(np.abs([max_y, min_y]))
        scaling_factor = np.min([width / 2 / max_width, height / 2 / max_height])
        shape_zero_mean.points *= scaling_factor
        shape_zero_mean.points = np.rint(shape_zero_mean.points).astype(int)

        return shape_zero_mean.translated_to_point(mean)

    def axis_means(self):
        """Generates a 2x1 vector with means for x and y coordinates"""
        return np.mean(self.points, axis=0)

    def translate_to_origin(self):
        self.points = self.points - self.axis_means()

    def translated_to_point(self, point: Union[Point, np.ndarray]) -> "Shape":
        """Returns a new shape with mean translated to arbitrary point"""
        return Shape(self.points + np.array(point))

    def translated_to_origin(self) -> "Shape":
        """Returns a new shape with points translated to origin"""
        # return Shape(self.points - self.axis_means())
        return self.translated_to_point(-self.axis_means())

    def get_orthogonal_vectors(
        self, with_tck: bool = False
    ) -> Union[List[Point], Tuple[List[Point], Tuple[Any]]]:
        """Returns the estimated orthogonal unit vectors of the shape"""
        x = self.x_vector()
        y = self.y_vector()
        per = len(self)
        x = np.append(x, x[0])
        y = np.append(y, y[0])

        # Calculate cubic spline interpolation
        tck, _ = interpolate.splprep([x, y], per=per)

        # The parametric variable goes from [0, 1[ (since the last
        # point is a repetition)
        u = np.arange(0, 1, 1 / per)

        # Get derivatives for each point
        deriv_vecs = np.array(interpolate.splev(u, tck, der=1))

        # Rotate each derivative 90 degrees
        M = np.array([[0, -1], [1, 0]])
        orth_vecs = np.matmul(M, deriv_vecs)

        # Tranpose matrix and create list of vectors
        transp_orth = np.transpose(orth_vecs)
        vector_list = [Point(*i) for i in transp_orth]

        # Normalize to 10 pixels size
        norms = [np.linalg.norm(v) for v in vector_list]

        normalized_list = [
            Point(*np.multiply(np.divide(v, n), 20))
            for (v, n) in zip(vector_list, norms)
        ]

        if with_tck:
            return normalized_list, tck

        return normalized_list

    @classmethod
    def apply_procrustes(cls, shapes):
        """Align a set of N shapes according to the Procrustes method"""
        # Translate all shapes to be centered at (0, 0)
        Shape.translate_all_to_origin(shapes)
        ref_shape, *_ = shapes
        # Rotate, scale and translate each shape to align with the first shape in the set
        cls.align_many(ref_shape, shapes)

    @classmethod
    def mean_from_many(cls, shapes):
        """Finds the mean of a set of shapes"""
        y = np.array([shape.points for shape in shapes])
        return cls(np.mean(y, axis=0))

    @classmethod
    def translate_all_to_origin(cls, shapes):
        """Applies the translate_to_origin method to a list of shapes"""
        for shape in shapes:
            shape.translate_to_origin()
        return shapes

    @classmethod
    def align_many(cls, ref_shape, shapes):
        """Rotate, scale and translate each shape to align with the ref_shape"""
        for shape in shapes:
            shape.align(ref_shape)
        return shapes

    @classmethod
    def from_vector(cls, vector: np.ndarray):
        return cls(np.reshape(vector, (-1, 2)))

    @classmethod
    def copy(cls, source):
        """Creates a shape with the same points as a source shape"""
        return cls(np.copy(source.points))

    @classmethod
    def from_file(cls, image_index: int, incisor_index: int) -> "Shape":
        filename = "./data/Landmarks/original/landmarks{:d}-{:d}.txt".format(
            image_index, incisor_index
        )
        with open(filename) as f:
            coordinates = [int(float(x)) for x in f.readlines()]
            coord_iter = iter(coordinates)
            points = [Point(*p) for p in zip(coord_iter, coord_iter)]
            return cls(points)
