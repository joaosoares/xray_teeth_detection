import cv2
import numpy as np


class GrayLevelProfile:
    """Represents a normalized set of points along an axis"""

    def __init__(self, samples):
        self.samples = self.normalize(samples)

    def normalize(self, samples):
        # TODO: Normalize grayscale level
        return samples

    @classmethod

    @classmethod
    def get_point_norm_gradient(cls, image, center_point, direction_vector,
                                half_sampling_size) -> np.ndarray:
        """Extracts the normalized gradient of a vector along the specified
        direction and centerpoint of an image.
        """
        rot_img, x_center, y_center = cls.rotate_and_center_image(
            image, center_point, direction_vector)
        # Create array spanning from (0, -k) to (0, k)
        samples = rot_img[x_center, (y_center - half_sampling_size):(
            y_center + half_sampling_size + 1)]
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
