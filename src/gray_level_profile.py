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
    def from_image(cls, image, center_point, direction_vector,
                   half_sampling_size):
        """Creates a gray level profile of an image on the specified location"""
        # Translate POI to origin and rotate around so direction vector is horizontal
        angle = np.degrees(
            -np.arctan2(direction_vector[1], direction_vector[0])
        )  # invert because np fn uses (1, 0) as ref
        print(angle)
        rows, cols = image.shape

        M = cv2.getRotationMatrix2D(center_point, angle, 1)
        rot_img = cv2.warpAffine(image, M, (cols, rows))

        x_center, y_center = (rows // 2, cols // 2)
        # Create array spanning from (0, -k) to (0, k)
        samples = rot_img[x_center, (y_center - half_sampling_size):(
            y_center + half_sampling_size + 1)]
        print(samples)
        glp = cls(samples)
        return (glp, rot_img)
