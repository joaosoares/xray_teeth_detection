import numpy as np
from shape import Shape
from active_shape_model import ActiveShapeModel

s1 = Shape([(1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (3, 3), (3, 2), (2, 2)])
s2 = Shape([(5, 2), (6, 3), (7, 4), (8, 3), (9, 2), (8, 1), (7, 0), (6, 1)])

ActiveShapeModel.from_shapes([s1, s2])