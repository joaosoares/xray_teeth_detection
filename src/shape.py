'''Procrustes method library'''

from itertools import chain
from typing import List, NewType, Tuple, Union

import cv2
import numpy as np

Point = NewType('Point', int)  # Point does something


class Shape:
    def __init__(self, points: Union[List[Tuple[Point, Point]], np.ndarray]):
        self.points = np.array(points)

    def __len__(self):
        return np.shape(self.points)[0]

    def norm(self):
        """Returns the norm of a vector containing all points"""
        return np.linalg.norm(self.points)

    # This function may have too much fluctuation
    def align(self, ref_shape):
        """
        Aligns shape to another reference shape

        Aligns by minimizing the weighted sum described in Eq. 2 of Cootes et al.
        """
        # a coefficient
        a = np.dot(
            np.reshape(self.points, (-1)), np.reshape(ref_shape.points,
                                                      (-1))) / (self.norm()**2)
        # b coefficient
        n = len(self)
        b_sum = 0
        for i in range(n):
            b_sum += (self.points[i][0] * ref_shape.points[i][1] -
                      ref_shape.points[i][0] * self.points[i][1])
        b = b_sum / (self.norm()**2)
        # s and theta coeffs
        s = np.sqrt(a**2 + b**2)
        theta = np.tanh(b / a)
        # rotation matrix
        rot_mat = np.array([(np.cos(theta), np.sin(theta)), (-np.sin(theta),
                                                             np.cos(theta))])
        # final point calculation
        self.points = s * np.matmul(self.points, rot_mat)

    def get_weights(self, points):
        pass

    def normalize(self):
        """Normalizes the shape to a default scale and pose"""
        pass

    def axis_means(self):
        """Generates a 2x1 vector with means for x and y coordinates"""
        return np.mean(self.points, axis=0)

    def translate_to_origin(self):
        self.points = self.points - self.axis_means()

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
        return cls(np.mean(y, axis=2))

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
