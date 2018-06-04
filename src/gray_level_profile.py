import cv2
import numpy as np

from typing import List, Tuple
from point import Point

from image_shape import ImageShape


class GrayLevelProfile:
    """Represents a normalized set of points along an axis"""

    def __init__(self, mean_profile, covariance):
        self.mean_profile = mean_profile
        self.covariance = covariance
        self.inv_cov = np.inv(self.covariance)

    def mahalanobis_distance(self, sample):
        """Calculates the Mahalanobis distance of the sample to the GLP"""
        error = sample - self.mean_profile
        return np.transpose(error) @ self.inv_cov @ error

    @classmethod
    def from_image_shapes(
        cls, images: List[ImageShape], points: List[Point], half_sampling_size: int = 8
    ):
        # numpy array to store all samples
        point_norm_gradients = np.zeros((len(images), half_sampling_size * 2 + 1))
        # iterate over each sample
        for idx, image_shape in enumerate(images):
            vectors = image_shape.shape.get_orthogonal_vectors()
            points = image_shape.shape.as_point_list()
            for point, vector in zip(points, vectors):
                point_norm_gradients[idx, :] = cls.get_point_norm_gradient(
                    point[0], point[1], vector, half_sampling_size
                )
        mean_vector = np.mean(point_norm_gradients, axis=0)
        covariance = np.cov(point_norm_gradients)
        return cls(mean_vector, covariance)

    @classmethod
    def get_point_norm_gradient(
        cls, image, center_point, direction_vector, half_sampling_size
    ) -> np.ndarray:
        """Extracts the normalized gradient of a vector along the specified
        direction and centerpoint of an image.
        """
        rot_img, x_center, y_center = cls.rotate_and_center_image(
            image, center_point, direction_vector
        )
        # Create array spanning from (0, -k) to (0, k)
        samples = rot_img[
            x_center,
            (y_center - half_sampling_size) : (y_center + half_sampling_size + 1),
        ]
        print(samples)

        # Take derivative and normalize
        norm_sample_gradients = cls.normalize(np.gradient(samples))
        return norm_sample_gradients

    @staticmethod
    def normalize(samples):
        """Normalizes sample by dividing through by sum of absolute element values"""
        return samples / sum(samples)

    @staticmethod
    def rotate_and_center_image(image, center_point, direction_vector):
        rows, cols = image.shape
        # Translate POI to origin and rotate around so direction vector is horizontal
        angle = np.degrees(
            -np.arctan2(direction_vector[1], direction_vector[0])
        )  # invert because np fn uses (1, 0) as ref

        # Create rotation matrix and apply warp affine
        M = cv2.getRotationMatrix2D(center_point, angle, 1)
        rot_img = cv2.warpAffine(image, M, (cols, rows))

        # Find new image center
        x_center, y_center = (rows // 2, cols // 2)

        return (rot_img, x_center, y_center)
