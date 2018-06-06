from enum import Enum

from active_shape_model import ActiveShapeModel
from image_shape import ImageShape
from point import Point
from shape import Shape


class Incisors(Enum):
    UPPER_OUTER_LEFT = 1
    UPPER_INNER_LEFT = 2
    UPPER_INNER_RIGHT = 3
    UPPER_OUTER_RIGHT = 4
    LOWER_OUTER_LEFT = 5
    LOWER_INNER_LEFT = 6
    LOWER_INNER_RIGHT = 7
    LOWER_OUTER_RIGHT = 8

    @classmethod
    def active_shape_models(cls, images, incisors=None):
        """
        Computes the ActiveShapeModels for each incisor. Returns two dicts,
        one with active shape models and the other with the corresponding
        image_shapes used for each incisor.
        """
        if incisors == None:
            incisors = cls

        active_shape_models = {}
        all_image_shapes = {}
        for incisor in incisors:
            image_shapes = [
                ImageShape(image, Shape.from_file(i, incisor.value))
                for i, image in enumerate(images, 1)
            ]
            all_image_shapes[incisor] = image_shapes
            active_shape_models[incisor] = ActiveShapeModel.from_image_shapes(
                image_shapes
            )
        return active_shape_models, all_image_shapes
