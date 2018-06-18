"""Module with class to evaluate obtained results"""
from typing import Type, Union, Text

from auto_initializator import AutoInitializator
from manual_initializator import ManualInitializator
from image_shape import ImageShape


class Evaluator:
    def __init__(self, auto_init: bool = False) -> None:
        self.initializator: Union[Type[AutoInitializator], Type[ManualInitializator]]
        if auto_init:
            self.initializator = AutoInitializator
        else:
            self.initializator = ManualInitializator

    def quantitative_eval(self):
        """Performs n-1 evaluation"""
        pass

    def mean_squared_error(self):
        """Calculates mean squarred error for an image shape."""
        pass

    def qualitative_eval(self):
        """Saves all image shapes so they can be compared"""
        pass

    def save_image_shape(self, image_shape: ImageShape, filename: Text) -> None:
        """Saves an image shape with a given filename"""
        pass
