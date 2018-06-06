# Assorted utilities for dealing with Shapes and ActiveShapeModels
import matplotlib.pyplot as plt
from matplotlib import collections
import numpy as np

from typing import List, Union

from shape import Shape
from point import Point


def plot_shape(shapes: Union[Shape, List[Shape]], overlay_image=None, plot=True):
    """Plots a single shape or an array of shapes"""
    if type(shapes) is Shape:
        shapes: List[Shape] = [shapes]

    for shape in shapes:
        # Append the first point in the end to draw line b/w 1st and last
        x_values = np.append(shape.points[:, 0], shape.points[:, 0][0])
        y_values = np.append(shape.points[:, 1], shape.points[:, 1][0])

        plt.plot(x_values, y_values, "-o")

    if overlay_image is not None:
        plt.imshow(overlay_image)

    plt.axes().set_aspect("equal", "datalim")
    if plot:
        plt.show()


def plot_vecs(
    vectors: Union[np.ndarray, List[np.ndarray]],
    points: Union[Point, List[Point]],
    plot=True,
):
    """Plots a vector or an array of vectors centered around a point or
    list of points."""
    if type(vectors) is np.ndarray:
        vectors = [vectors]
    if type(points) is tuple:
        points = [points]

    if len(vectors) != len(points):
        raise ValueError("Number of vectors is different from number of points")

    lines = [
        [point, tuple(map(sum, zip(point, vector)))]
        for vector, point in zip(vectors, points)
    ]
    print(lines)
    lc = collections.LineCollection(lines)

    plt.axes().add_collection(lc)
    plt.axes().set_aspect("equal", "datalim")

    if plot:
        plt.show()
