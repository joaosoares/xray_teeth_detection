import sys
from itertools import chain
from typing import List

import cv2
import numpy as np

from image_shape import ImageShape
from point import Point

from shapeutils import plot_shape


class GrayLevelProfile:
    """Represents a normalized set of points along an axis"""

    def __init__(
        self, mean_profile: np.ndarray, covariance: np.ndarray, half_sampling_size: int
    ) -> None:
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

    @staticmethod
    def point_pos_from_profiles_list(
        profile_index: int, profiles_count: int, original_point: Point, direction_vector
    ) -> Point:
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
        rev_angle = np.arctan2(direction_vector[0], direction_vector[1])

        # define vector from origin to new point on axis
        axis_vec = Point(profile_index, 0)

        # Rotate this vector to get necessary image displacement
        cosine, sine = np.cos(rev_angle), np.sin(rev_angle)
        rot_mat = np.array(((cosine, -sine), (sine, cosine)))
        result = rot_mat @ np.array(axis_vec)
        img_vec = Point(result[1], result[0]).round()

        # Add img_vec to original point to get new point
        return original_point - img_vec

    @staticmethod
    def point_pos_from_single_profile(
        elem_idx: int, elem_count: int, original_point: Point, direction_vector: Point
    ) -> Point:
        # Adjust index to make center point be zero
        k = (elem_count - 1) // 2
        elem_idx = elem_idx - k

        # on rotate_and_center_image, there is a minus sign before np.arctan2
        rev_angle = np.arctan2(direction_vector[0], direction_vector[1])

        # define vector from origin to new point on axis
        axis_vec = Point(elem_idx, 0)

        # Rotate this vector to get necessary image displacement
        cosine, sine = np.cos(rev_angle), np.sin(rev_angle)
        rot_mat = np.array(((cosine, -sine), (sine, cosine)))
        result = rot_mat @ np.array(axis_vec)
        img_vec = Point(result[1], result[0]).round()

        # Add img_vec to original point to get new point
        return original_point - img_vec

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
        for points_samples in images_samples:
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
        processing_fn=None,
    ) -> List[np.ndarray]:
        """Extracts the normalized gradient of a vector along the specified
        direction and centerpoint of an image.
        """
        # Set default sliding_profiles_count
        if profile_count is None:
            profile_count = half_sampling_size

        if processing_fn is None:
            processing_fn = cls.normalize

        # Rotate image so that horizontal line is aligned to direction vector
        rot_img = cls.rotate_and_center_image(image, center_point, direction_vector)

        x_center, y_center = center_point.round()
        # Create 2k samples spanning from -2k to 2k with lengths 2k+1
        y_values = [
            (
                (y_center + i) - half_sampling_size,
                (y_center + i) + half_sampling_size + 1,
            )
            # pylint: disable=E1130
            # (We are already checking against it being None)
            for i in chain(range(profile_count + 1), range(-profile_count, 0))
        ]
        samples = np.array(
            [rot_img[y_value[0] : y_value[1], x_center] for y_value in y_values]
        )
        normalized_samples = processing_fn(samples)

        return normalized_samples

    @staticmethod
    def normalize(samples):
        """Normalizes sample by dividing through by sum of absolute element values"""
        summed = np.sum(np.abs(samples), axis=1)
        if summed.any() != 0:
            normalized = samples / summed.reshape((-1, 1))
        else:
            normalized = samples
        return normalized

    @staticmethod
    def rotate_and_center_image(image, center_point, direction_vector) -> np.ndarray:
        rows = image.shape[0]
        cols = image.shape[1]
        # Translate POI to origin and rotate around so direction vector is horizontal
        angle = np.degrees(np.arctan2(direction_vector[0], direction_vector[1]))

        # Create rotation matrix and apply warp affine
        rot_mat = cv2.getRotationMatrix2D(center_point, angle, 1)
        rot_img = cv2.warpAffine(image, rot_mat, (cols, rows), flags=cv2.INTER_LINEAR)

        return rot_img
