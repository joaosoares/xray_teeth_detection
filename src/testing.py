import numpy as np
from shape import Shape

shapes = [
    Shape([(1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (3, 3), (3, 2), (2, 2)]),
    Shape([(11, 2), (11, 3), (11, 4), (12, 4), (13, 4), (13, 3), (13, 2),
           (12, 2)]),
]

# for shape_obj in shapes:
#     print(len(shape_obj))

# Shape.translate_all_to_origin(shapes)

# for shape_obj in shapes:
#     print(shape_obj.norm())

s1 = Shape([(1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (3, 3), (3, 2), (2, 2)])
s2 = Shape([(11, 2), (11, 3), (11, 4), (12, 4), (13, 4), (13, 3), (13, 2),
            (12, 2)])
s3 = Shape([(2, 4), (2, 6), (2, 8), (4, 8), (6, 8), (6, 6), (6, 4), (4, 4)])
s4 = Shape([(5, 2), (6, 3), (7, 4), (8, 3), (9, 2), (8, 1), (7, 0), (6, 1)])

Shape.translate_all_to_origin([s1, s2, s3, s4])

print("Points after translation")
print(s1.points)
print(s4.points)

print("Alignment")
s4.align(s1)

print(s1.points)
print(s4.points)