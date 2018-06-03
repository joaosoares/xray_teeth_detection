from enum import Enum

import cv2
import matplotlib.pyplot as plt
import numpy as np

from active_shape_model import ActiveShapeModel
from image_shape import ImageShape
from point import Point
from shape import Shape
from shapeutils import plot_shape


class Incisors(Enum):
    UPPER_OUTER_LEFT = 1
    UPPER_INNER_LEFT = 2
    UPPER_INNER_RIGHT = 3
    UPPER_OUTER_RIGHT = 4
    LOWER_OUTER_LEFT = 5
    LOWER_INNER_LEFT = 6
    LOWER_INNER_RIGHT = 7
    LOWER_OUTER_RIGHT = 8


def get_shape_from_file(image_index: int, incisor_index: int) -> Shape:
    filename = './data/Landmarks/original/{%2d}-{%d}'.format(
        image_index, incisor_index)
    with open(filename) as f:
        coordinates = [int(float(x)) for x in f.readlines()]
        coord_iter = iter(coordinates)
        points = [Point(*p) for p in zip(coord_iter, coord_iter)]
        return Shape(points)


def main():
    # Import all training images
    image_filenames = [
        "./data/Radiographs/{%2d}".format(i) for i in range(1, 15)
    ]
    images = [cv2.imread(filename) for filename in image_filenames]

    active_shape_models = {}
    # For each out of the 8 shapes, create imageshape for all training images
    for incisor in Incisors:

        image_shapes = [
            ImageShape(image, get_shape_from_file(i, incisor))
            for image, i in enumerate(images, 1)
        ]
        active_shape_models[incisor] = ActiveShapeModel.from_image_shapes(
            image_shapes)

    # For each training image, import its landmarks and
    # base_path = './data/Radiographs/'
    landmarks_path = './data/Landmarks/original/'
    # cur_img = cv2.imread(base_path + '01.tif')
    teeth_points = []
    for i in range(1, 15):
        teeth_points.append(
            get_landmark_pairs(landmarks_path + "landmarks{}-1.txt".format(i)))

    tooth_shapes = [Shape(tooth_points) for tooth_points in teeth_points]

    Shape.apply_procrustes(tooth_shapes)

    am = ActiveShapeModel.from_imageshapes(tooth_shapes)

    shapes = []
    for b in np.arange(-0.1 * am.eigenvalues[0], 0.1 * am.eigenvalues[0],
                       0.2 * am.eigenvalues[0] / 4):
        # bs = [0, 1, 2, 3, 5]
        # for b in bs:
        shape_params = np.zeros(len(am))
        shape_params[0] = b
        print(shape_params)
        shapes.append(am.create_shape(shape_params))
    plot_shape(shapes)

    # for tooth in teeth:
    #     plt.plot(tooth.points[:, 0], tooth.points[:, 1])
    # plt.show()


if __name__ == '__main__':
    main()
