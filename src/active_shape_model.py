# ActiveShapeModel class

from typing import List, NewType, Tuple
from sklearn.decomposition import PCA

import numpy as np

from shape import Shape


class ActiveShapeModel:
    def __init__(self, param_toler_pairs):
        self.points = points

    @classmethod
    def from_shapes(cls, shapes, des_expvar_ratio=0.95):
        # Find mean shape
        mean_shape = Shape.mean_from_many(shapes)
        # Create matrix from all shapes
        shape_mat = np.array(
            [np.reshape(shape.points, (-1)) for shape in shapes])

        pca = PCA(2)
        res = pca.fit_transform(shape_mat)

        # Compute covariance matrix
        cov_mat = np.cov(np.transpose(shape_mat))
        # Eigenvalues and eigenvectors of covariance matrix
        e_values, e_vectors = np.linalg.eig(cov_mat)

        sort_idx = np.argsort(-e_values)

        selected_evectors = []
        cur_expvar_ratio = 0
        cur_idx = 0
        while (cur_expvar_ratio < des_expvar_ratio) and (cur_idx <
                                                         len(shapes)):
            actual_idx = sort_idx[cur_idx]
            selected_evectors.append(e_vectors[actual_idx])
            cur_idx += 1
            cur_expvar_ratio += np.abs(e_values[actual_idx]) / np.abs(
                sum(e_values))

        parameter_count = cur_idx

        return
