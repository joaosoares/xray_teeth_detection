# ActiveShapeModel class

from typing import List, Tuple

import numpy as np
from sklearn.decomposition import PCA

from shape import Shape
import shapeutils as util


class ActiveShapeModel:
    """
    ActiveShapeModel represents an instance of an ActiveShape that can be
    used to fit a new Shape to.
    """

    def __init__(self, mean_shape: Shape, eigenvectors: np.ndarray,
                 eigenvalues: np.ndarray):
        self.mean_shape = mean_shape
        self.eigenvectors = np.real(eigenvectors)
        self.eigenvalues = np.real(eigenvalues)
        self.origin = (0, 0)

    def __len__(self):
        """
        Returns the number of parameters in the b_vector,
        which is the same as the number of eigenvectors
        """
        return self.eigenvectors.shape[0]

    def match_target(self, target_shape: Shape):
        """Matches an active model to a target Shape"""
        shape_params = np.zeros(len(self))

        converged = False
        all_est_shapes = []
        while not converged:
            # 2. Generate the model point positions using x = x_mean + P*b
            est_shape = self.create_shape(shape_params)
            all_est_shapes.append(est_shape)

            # Apply Procrustes method to align initial estimation to target shape
            # Project target shape Y into the model coordinate frame by inverting the transformation T
            target_shape_copy = Shape.copy(target_shape)
            target_shape_copy.align(est_shape)

            # 5. Project y into the tangent plane to x_mean by scaling y' = y/(y * x_mean)
            target_shape_copy.points = target_shape_copy.points / np.dot(
                target_shape_copy.as_vector(), self.mean_shape.as_vector())

            # 6. Update the model parameters to match to y'
            prev_shape_params = shape_params
            shape_params = self.update_shape_parameters(target_shape_copy)

            # 7. If not converged, return to step 2
            converged = self.check_convergence(shape_params, prev_shape_params)

        util.plot_shape(all_est_shapes)
        return est_shape

    @staticmethod
    def check_convergence(shape_params, prev_shape_params, max_delta=0.000001):
        """
        Compares two sets of shape parameters and returns a boolean indicating
        whether their biggest difference is smaller than the maximum delta
        """
        difference_array = np.abs(shape_params - prev_shape_params)
        return np.max(difference_array) < max_delta

    def create_shape(self, shape_parameters: np.ndarray):
        """Creates a shape from a set of b_parameters"""
        if shape_parameters.shape != (len(self), ):
            raise ValueError(
                "Vector with B parameters is of size {} but expected size {}".
                format(shape_parameters.shape, (len(self), )))

        mu = self.mean_shape.as_vector()
        P_b = np.matmul(np.transpose(self.eigenvectors), shape_parameters)
        new_shape_points = mu + P_b
        return Shape.from_vector(new_shape_points)

    def update_shape_parameters(self, target_shape: Shape) -> np.ndarray:
        return np.matmul(
            target_shape.as_vector() - self.mean_shape.as_vector(),
            np.transpose(self.eigenvectors))

    @classmethod
    def from_shapes(cls, shapes: List[Shape], des_expvar_ratio=0.95):
        """
        Creates an ActiveShapeModel instance by finding the mean of
        a list of Shapes, then applying PCA to the list to derive the
        parameters with highest variance. Finally, we filter the eigenvalues
        by order of variance until we reach the desired explained
        variance.
        """
        # Find mean shape
        mean_shape = Shape.mean_from_many(shapes)
        mean_shape.translate_to_origin()
        # Create matrix from all shapes
        shape_mat = np.array(
            [np.reshape(shape.points, (-1)) for shape in shapes])

        # Compute covariance matrix
        cov_mat = np.cov(np.transpose(shape_mat))
        # Eigenvalues and eigenvectors of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov_mat)

        pca_obj = PCA()
        pca_obj.fit(shape_mat)

        # # Sort
        # sorted_indices = np.argsort(-eigenvalues)

        # # Select eigenvectors according to eigenvalue sort and precision
        # selected_eigenvectors = np.array([
        #     evec for idx, evec in enumerate(eigenvectors[sorted_indices])
        #     if sum(eigenvalues[sorted_indices[:idx]]) /
        #     sum(eigenvalues) < des_expvar_ratio
        # ])

        # # Select eigenvalues according to eigenvalue sort and precision
        # selected_eigenvalues = np.array([
        #     evalue for idx, evalue in enumerate(eigenvalues[sorted_indices])
        #     if sum(eigenvalues[sorted_indices[:idx]]) /
        #     sum(eigenvalues) < des_expvar_ratio
        # ])

        return cls(mean_shape, pca_obj.components_,
                   pca_obj.explained_variance_)
