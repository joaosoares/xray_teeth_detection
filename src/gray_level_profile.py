import sys
from itertools import chain
from typing import List, Tuple

import cv2
import numpy as np

from image_shape import ImageShape
from shape import Shape
from point import Point

from shapeutils import plot_shape


class GrayLevelProfile:
    """Represents a normalized set of points along an axis"""

    def __init__(self, mean_profile, covariance, half_sampling_size):
        self.mean_profile = mean_profile
        self.covariance = np.matrix(covariance)
        self.half_sampling_size = half_sampling_size
        # calculate inverse of covariance
        if np.linalg.cond(self.covariance) < 1 / sys.float_info.epsilon:
            print("singular matrix")
        self.inv_cov = self.covariance.I

    def mahalanobis_distance(self, sample: np.ndarray):
        """Calculates the Mahalanobis distance of the sample to the GLP"""
        error = sample - self.mean_profile
        return (error.T @ self.inv_cov @ error).item(0, 0)

    def get_point_position(
        self,
        profile_index: int,
        profiles_count: int,
        original_point: Point,
        direction_vector,
    ):
        """
        Given the index of a profile, figure out approximately which
        point of the original image corresponds to it.
        """
        # profiles_count is always odd
        max_right_idx = (profiles_count - 1) / 2

        # adjust to get values to the left of the original point
        if profile_index > max_right_idx:
            profile_index -= profiles_count

        # on rotate_and_center_image, there is a minus sign before np.arctan2
        rev_angle = np.degrees(np.arctan2(direction_vector[1], direction_vector[0]))

        # define vector from origin to new point on axis
        axis_vec = Point(profile_index, 0)

        # Rotate this vector to get necessary image displacement
        c, s = np.cos(rev_angle), np.sin(rev_angle)
        rot_mat = np.array(((c, -s), (s, c)))
        result = rot_mat @ np.array(axis_vec)
        img_vec = Point(*[int(p) for p in result])

        # Add img_vec to original point to get new point
        return original_point + img_vec

    @classmethod
    def from_image_shapes(cls, images: List[ImageShape], half_sampling_size: int = 20):
        """Creates a GLP for each landmark point using the shapes and images provided."""
        images_samples: List[List[np.ndarray]] = [
            [] for i in range(len(images[0].shape))
        ]

        # iterate over each sample
        for image_shape in images:
            vectors = image_shape.shape.get_orthogonal_vectors()
            points = image_shape.shape.as_point_list()
            for idx, (point, vector) in enumerate(zip(points, vectors)):
                sample = cls.sliding_profiles(
                    image_shape.image, point, vector, half_sampling_size, 0
                )[0]
                images_samples[idx].append(sample)

        profiles = []
        for i, points_samples in enumerate(images_samples):
            print(f"Point {i}")
            points_samples_arr = np.array(points_samples)
            mean_vector = np.mean(points_samples_arr, axis=0)
            covariance = np.cov(points_samples_arr.T)
            profiles.append(cls(mean_vector, covariance, half_sampling_size))

        return profiles

    @classmethod
    def sliding_profiles(
        cls,
        image,
        center_point,
        direction_vector,
        half_sampling_size,
        profile_count=None,
    ) -> List[np.ndarray]:
        """Extracts the normalized gradient of a vector along the specified
        direction and centerpoint of an image.
        """
        # Set default sliding_profiles_count
        if profile_count == None:
            profile_count = half_sampling_size

        # Rotate image so that horizontal line is aligned to direction vector
        rot_img = cls.rotate_and_center_image(image, center_point, direction_vector)

        x_center, y_center = center_point

        # Create 2k samples spanning from -2k to 2k with lengths 2k+1
        y_values = [
            (
                (y_center + i) - half_sampling_size,
                (y_center + i) + half_sampling_size + 1,
            )
            for i in chain(range(profile_count + 1), range(-profile_count, 0))
        ]
        samples = [rot_img[y_value[0] : y_value[1], x_center] for y_value in y_values]
        normalized_samples = [cls.normalize(np.gradient(sample)) for sample in samples]

        # first
        # sample_points = rot_img[
        #     x_center,
        #     (y_center - half_sampling_size) : (y_center + half_sampling_size + 1),
        # ]
        # print(sample_points)

        # Take derivative and normalize
        # norm_sample_gradients = cls.normalize(np.gradient(sample_points))
        return normalized_samples

    @staticmethod
    def normalize(samples):
        """Normalizes sample by dividing through by sum of absolute element values"""
        summed = sum(np.abs(samples))
        if summed != 0:
            normalized = samples / summed
        else:
            normalized = samples
        return normalized

    @staticmethod
    def rotate_and_center_image(image, center_point, direction_vector) -> np.ndarray:
        rows = image.shape[0]
        cols = image.shape[1]
        # Translate POI to origin and rotate around so direction vector is horizontal
        angle = np.degrees(
            -np.arctan2(direction_vector[1], direction_vector[0])
        )  # invert because np fn uses (1, 0) as ref

        # Create rotation matrix and apply warp affine
        M = cv2.getRotationMatrix2D(center_point, angle, 1)

        rot_img = cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_NEAREST)

        return rot_img
