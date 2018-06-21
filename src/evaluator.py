"""Module with class to evaluate obtained results"""
from typing import Text, Type, Union, Dict, List, cast

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

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
        printable: ImageShapesDict = None,
    ) -> None:
        self.initial = initial
        self.expected = expected
        if printable is None:
            self.printable = expected
        else:
            self.printable = printable
        self.actual: ImageShapesDict = {}

    def quantitative_eval(self):
        """Performs leave-one-out evaluation"""
        root_mean_squared_errors = {}
        for incisor in Incisors:
            initial = self.expected[incisor]
            expected = self.expected[incisor]
            printable = self.printable[incisor]
            self.actual[incisor] = []
            root_mean_squared_errors[incisor] = []
            print(incisor)
            for i, (initial_imgshp, expected_imgshp, printable_imgshp) in enumerate(
                zip(initial, expected, printable)
            ):
                # Create array with all other expected image shapes
                other_image_shapes = [
                    imgshp for imgshp in expected if (imgshp is not expected_imgshp)
                ]
                # Find the ASM
                asm = ActiveShapeModel.from_image_shapes(other_image_shapes)

                # Apply found ASM of initial imageshape
                actual_image_shape = asm.fit_to_image(initial_imgshp)
                self.actual[incisor].append(actual_image_shape)

                # Calculate mean_squared error
                rmse = self.root_mean_squared_error(actual_image_shape, expected_imgshp)
                print(rmse)
                root_mean_squared_errors[incisor].append(rmse)

                # Save image
                self.save_image_shape(
                    ImageShape(printable_imgshp.image, actual_image_shape.shape),
                    f"./actual-{incisor}-{i}",
                )

    def root_mean_squared_error(
        self, actual: ImageShape, expected: ImageShape
    ) -> float:
        """Calculates mean squarred error for an image shape."""
        return np.sqrt(mean_squared_error(actual.shape.points, expected.shape.points))

    def qualitative_eval(self):
        """Saves all image shapes so they can be compared. Must be ran after
        quantitative eval"""
        for incisor in Incisors:
            for i, imgshp in enumerate(self.actual):
                filename = f"./actual-{i}-{incisor}"
                self.save_image_shape(imgshp, filename)

    def save_image_shape(self, image_shape: ImageShape, filename: Text) -> None:
        """Saves an image shape with a given filename"""
        plot_image_shape(
            image_shape, display=False, dots=False, interpol=False, vecs=False
        )
        plt.savefig(filename + ".png", dpi=600)
        plt.clf()
