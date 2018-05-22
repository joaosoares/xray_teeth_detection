# Assorted utilities for dealing with Shapes and ActiveShapeModels
import matplotlib.pyplot as plt
import numpy as np

from typing import List, Union

from active_shape_model import ActiveShapeModel
from shape import Shape


def plot_shape(shapes: Union[Shape, List[Shape]]):
    """Plots a single shape or an array of shapes"""
    if type(shapes) is Shape:
        shapes: List[Shape] = [shapes]

    max_abs_x = 1
    max_abs_y = 1
    for shape in shapes:
        # Append the first point in the end to draw line b/w 1st and last
        x_values = np.append(shape.points[:, 0], shape.points[:, 0][0])
        y_values = np.append(shape.points[:, 1], shape.points[:, 1][0])

        plt.plot(x_values, y_values, '-o')
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()