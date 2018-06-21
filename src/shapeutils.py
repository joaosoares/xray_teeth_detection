# Assorted utilities for dealing with Shapes and ActiveShapeModels
from typing import cast, Any, List, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib import collections
from scipy import interpolate

from point import Point
from shape import Shape
from image_shape import ImageShape


def plot_shape(
    shapes: Union[Shape, List[Shape]], overlay_image=None, display=True, dots=True
):
    """Plots a single shape or an array of shapes"""
    if isinstance(shapes, Shape):
        shapes = [shapes]

    for shape in shapes:
        # Append the first point in the end to draw line b/w 1st and last
        x_values = np.append(shape.points[:, 0], shape.points[:, 0][0])
        y_values = np.append(shape.points[:, 1], shape.points[:, 1][0])
        if dots:
            plt.plot(x_values, y_values, "-o")
        else:
            plt.plot(x_values, y_values, "-")

    if overlay_image is not None:
        plt.imshow(overlay_image, cmap="gray")

    plt.axes().set_aspect("equal", "datalim")
    if display:
        plt.show()


def plot_point(points: Union[Point, List[Point]], display=True):
    """Plots a single point or a list of points"""
    if isinstance(points, Point):
        points = [points]

    for point in points:
        plt.plot(point.x, point.y, "-o")
    plt.axes().set_aspect("equal", "datalim")
    if display:
        plt.show()


def plot_vecs(
    vectors: Union[np.ndarray, List[np.ndarray]],
    points: Union[Point, List[Point]],
    display=True,
):
    """Plots a vector or an array of vectors centered around a point or
    list of points."""
    if isinstance(vectors, (np.ndarray, Point)):
        vectors = [vectors]
    if isinstance(points, (tuple, Point)):
        points = [points]

    if len(vectors) != len(points):
        raise ValueError("Number of vectors is different from number of points")

    lines = [[point - vector, point + vector] for vector, point in zip(vectors, points)]
    lc = collections.LineCollection(np.array(lines))

    plt.axes().add_collection(lc)
    plt.axes().set_aspect("equal", "datalim")

    if display:
        plt.show()


def plot_interpol(tck: Tuple[Any], display=True):
    u = np.arange(0, 1.01, 0.01)
    out = interpolate.splev(u, tck)

    plt.plot(out[0], out[1])
    if display:
        plt.show()


def plot_rectangle(bottom_left, top_right, display=True):
    width, height = top_right - bottom_left
    p = patches.Rectangle(bottom_left, width, height, fill=False)
    plt.axes().add_patch(p)
    plt.axes().set_aspect("equal", "datalim")

    if display:
        plt.show()


def plot_profile(profiles: Union[np.ndarray, List[np.ndarray]], display=True):
    if isinstance(profiles, np.ndarray):
        profiles = [profiles]

    for i, profile in enumerate(profiles):
        plt.figure()
        plt.imshow(profile)

    if display:
        plt.show()


def plot_image_shape(
    image_shapes: Union[ImageShape, List[ImageShape]],
    display=True,
    dots=True,
    interpol=True,
    vecs=True,
):
    if isinstance(image_shapes, ImageShape):
        image_shapes = [image_shapes]

    for image_shape in image_shapes:
        shape = image_shape.shape
        image = image_shape.image
        points = shape.as_point_list()
        vectors, interpol = shape.get_orthogonal_vectors(with_tck=True)
        plot_shape(shape, display=False, overlay_image=image, dots=dots)
        if vecs:
            plot_vecs(vectors, points, display=False)

        if interpol:
            interpol = cast(Tuple[Any], interpol)
            plot_interpol(interpol, display=False)
    if display:
        plt.show()
