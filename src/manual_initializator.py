from typing import Dict, List, NewType, NamedTuple

from numpy import np

from active_shape_model import ActiveShapeModel
from image_shape import ImageShape
from incisors import Incisors
from point import Point

AsmDict = NewType("AsmDict", Dict[Incisors, ActiveShapeModel])
InitCoordsDict = NewType("InitCoordsDict", Dict[Incisors, List[InitCoord]])


class InitCoord(NamedTuple):
    """Initial coordinates"""

    bottom_left: Point
    top_right: Point


class ManualInitializator:
    @staticmethod
    def initialize(asms: AsmDict, images: np.ndarray[int]) -> List[ImageShape]:
        for image in images:
            for incisor in Incisors:
                pass
