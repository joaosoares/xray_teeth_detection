from typing import NamedTuple

import numpy as np


# Represents a point of the format (x, y), refering to the index
# of an np.ndarray
class Point(NamedTuple):
    """Represents a point with x and y coords"""

    x: int
    y: int

    def add_noise(self):
        """Returns a new point with random added noise"""
        p_with_noise = Point(
            int(self.x + np.random.normal(scale=5.0)),
            int(self.y + np.random.normal(scale=5.0)),
        )
        return p_with_noise

    def round(self):
        """Returns a new point with the closest integer coordinates"""
        rounded_p = Point(int(round(self.x)), int(round(self.y)))
        return rounded_p

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __truediv__(self, other):
        if isinstance(other, Point):
            return Point(self.x / other.x, self.y / other.y)
        return Point(self.x / other, self.y / other)
