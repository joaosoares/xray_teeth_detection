import numpy as np

import shapeutils as util
from active_shape_model import ActiveShapeModel
from shape import Shape

s1 = Shape([(1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (3, 3), (3, 2), (2, 2)])
s2 = Shape([(5, 2), (6, 3), (7, 4), (8, 3), (9, 2), (8, 1), (7, 0), (6, 1)])
s3 = Shape([(5, 2), (6, 3), (8, 4), (8, 3), (9, 2), (8, 1), (7, 0), (6, 1)])
s4 = Shape([(5, 2), (6, 3), (6, 4), (8, 3), (9, 2), (8, 1), (7, 0), (6, 1)])
s5 = Shape([(5, 2), (6, 2), (8, 4), (8, 3), (9, 2), (8, 1), (7, 0), (6, 1)])
s6 = Shape([(5, 2), (6, 4), (8, 4), (8, 3), (9, 2), (8, 1), (7, 0), (6, 1)])

se1 = Shape([(1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3), (1, 2)])
se2 = Shape([(1, 1), (2, 1), (3, 1), (3, 2), (3, 4), (2, 3.5), (1, 3), (1, 2)])

am = ActiveShapeModel.from_shapes([se1, se2])

util.plot_shape([se1, se2] + [am.mean_shape] +
                [am.create_shape(am.eigenvalues)])

# Estimated shape 1
# es1 = am.match_target(s1)

# shapes = []
# for b in np.arange(-0.3*am.eigenvalues[0], 0.3*am.eigenvalues[0], 0.1*am.eigenvalues[0]/5):
#     shape_params = np.zeros(len(am))
#     shape_params[0] = b
#     print(shape_params)
#     shapes.append(am.create_shape(shape_params))

# Shape.translate_all_to_origin([s1, s4])
# util.plot_shape(shapes)
