"""Module with class to evaluate obtained results"""
from typing import Text, Type, Union, Dict, List, cast

import matplotlib.pyplot as plt
import numpy as np

from auto_initializator import AutoInitializator
from image_shape import ImageShape
from manual_initializator import ManualInitializator
from shapeutils import plot_image_shape

from active_shape_model import ActiveShapeModel
from incisors import Incisors

ImageShapesDict = Dict[Incisors, List[ImageShape]]


class Evaluator:
    def __init__(
        self,
        initial: ImageShapesDict,
        expected: ImageShapesDict,
        auto_init: bool = False,
    ) -> None:
        self.initializator: Union[Type[AutoInitializator], Type[ManualInitializator]]
        if auto_init:
            self.initializator = AutoInitializator
        else:
            self.initializator = ManualInitializator
        self.initial = initial
        self.expected = expected
        self.actual: ImageShapesDict = {[] for incisor in Incisors}

    def quantitative_eval(self):
        """Performs leave-one-out evaluation"""
        mean_squared_errors = {[] for incisor in Incisors}
        for incisor in Incisors:
            initial = self.expected[incisor]
            expected = self.expected[incisor]
            for initial_imgshp, expected_imgshp in zip(initial, expected):
                # Create array with all other expected image shapes
                other_image_shapes = [
                    imgshp for imgshp in expected if (imgshp != expected_imgshp)
                ]
                # Find the ASM
                asm = ActiveShapeModel.from_image_shapes(other_image_shapes)

                # Apply found ASM ot initial imageshape
                actual_image_shape = asm.fit_to_image(initial_imgshp)
                self.actual[incisor].append(actual_image_shape)

                # Calculate mean_squared error
                mse = self.mean_squared_error(actual_image_shape, expected_imgshp)
                mean_squared_errors[incisor].append(mse)

    def mean_squared_error(self, actual: ImageShape, expected: ImageShape) -> float:
        """Calculates mean squarred error for an image shape."""

    def qualitative_eval(self):
        """Saves all image shapes so they can be compared. Must be ran after
        quantitative eval"""
        for incisor in Incisors:
            for i, imgshp in enumerate(self.actual):
                filename = f"./actual-{i}-{incisor}"
                self.save_image_shape(imgshp, filename)

    def save_image_shape(self, image_shape: ImageShape, filename: Text) -> None:
        """Saves an image shape with a given filename"""
        plot_image_shape(image_shape, display=False)
        plt.savefig(filename + ".png")
