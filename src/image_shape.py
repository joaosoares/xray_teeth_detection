from typing import NamedTuple

import numpy as np

from shape import Shape


class ImageShape(NamedTuple):
    image: np.ndarray
    shape: Shape