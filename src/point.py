from typing import NamedTuple

import numpy as np


# Represents a point of the format (x, y), refering to the index
# of an np.ndarray
class Point(NamedTuple):
    x: int
    y: int

    @classmethod
    def add_noise(cls, p):
        p_with_noise = cls(
            int(p.x + np.random.normal(scale=5.0)),
            int(p.y + np.random.normal(scale=5.0)),
        )
        return p_with_noise

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
