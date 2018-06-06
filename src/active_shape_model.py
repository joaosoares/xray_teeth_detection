# ActiveShapeModel class

import operator
from typing import List, Tuple

import numpy as np
from sklearn.decomposition import PCA

import shapeutils as util
from gray_level_profile import GrayLevelProfile
from image_shape import ImageShape
from shape import Shape


class ActiveShapeModel:
    """
    ActiveShapeModel represents an instance of an ActiveShape that can be
    used to fit a new Shape to.
    """

    mean_shape: Shape
    eigenvectors: np.ndarray
    eigenvalues: np.ndarray
    profiles: List[GrayLevelProfile]

    def __init__(
        self,
        mean_shape: Shape,
        eigenvectors: np.ndarray,
        eigenvalues: np.ndarray,
        profiles: List[GrayLevelProfile],
    ):
        self.mean_shape = mean_shape
        self.eigenvectors = np.real(eigenvectors)
        self.eigenvalues = np.real(eigenvalues)
        self.gray_level_profiles = profiles

    def __len__(self):
        """
        Returns the number of parameters in the b_vector,
        which is the same as the number of eigenvectors
        """
        return self.eigenvectors.shape[0]

    def match_target(self, target_shape: Shape, shape_params=None):
        """Matches an active model to a target Shape"""
        if not shape_params:
            shape_params = np.zeros(len(self))

        converged = False
        all_est_shapes = []
        while not converged:
            # Generate the model point positions using x = x_mean + P*b
            est_shape = self.create_shape(shape_params)
            all_est_shapes.append(est_shape)

            # Apply Procrustes method to align initial estimation to target shape
            target_shape_copy = Shape.copy(target_shape)
            target_shape_copy.align(est_shape)

            # Update the model parameters to match to aligned target_shape
            prev_shape_params = shape_params
            shape_params = self.update_shape_parameters(target_shape_copy)

            # If not converged, rerun the loop
            converged = self.check_convergence(shape_params, prev_shape_params)

        # Align estimated shape to the original target_shape to move it in the plane
        est_shape.align(target_shape)
        return est_shape, shape_params

    def create_shape(self, shape_parameters: np.ndarray):
        """Creates a shape from a set of b_parameters"""
        if shape_parameters.shape != (len(self),):
            raise ValueError(
                "Vector with B parameters is of size {} but expected size {}".format(
                    shape_parameters.shape, (len(self),)
                )
            )
        mu = self.mean_shape.as_vector()
        P_b = np.matmul(np.transpose(self.eigenvectors), shape_parameters)
        new_shape_points = mu + P_b
        return Shape.from_vector(new_shape_points)

    def update_shape_parameters(self, target_shape: Shape) -> np.ndarray:
        """Updates shape parameters to match target_shape"""
        new_shape_parameters = np.matmul(
            target_shape.as_vector() - self.mean_shape.as_vector(),
            np.transpose(self.eigenvectors),
        )
        limits = 3 * np.sqrt(self.eigenvalues)
        limited_shape_parameters = np.clip(new_shape_parameters, -limits, limits)
        return limited_shape_parameters

    def fit_to_image(self, image, search_image, inital_estimate):
        """
        Receives an image and iterates the initial_estimate to accurately
        fit the model in a region of that image.
        """
        # Initialize b=0, x=u
        shape_params = np.zeros(len(self))
        initial_estimate = self.mean_shape
        # Start iterating
        converged = False
        est_shape_history = []
        while not converged:
            # Search around each xi for best nearby image point yi
            suggested_shape = self.find_best_nearby_image_points(
                search_image, initial_estimate
            )
            # Save as current estimated_shape
            estimated_shape, shape_params = self.match_target(
                suggested_shape, shape_params
            )
            est_shape_history.append(estimated_shape)
            converged = self.check_convergence(
                *[shape.points for shape in est_shape_history[-2:-1]]
            )

    def find_best_nearby_image_points(self, search_image, shape):
        """
        Returns new shape where points are shifted to better locations
        according to the search_image information
        """
        # TODO: implement best nearby image points function
        pass

    @staticmethod
    def check_convergence(array_1, array_2, max_delta=0.001):
        """
        Compares two numpy arrays and returns a boolean indicating
        whether their biggest difference is smaller than the maximum delta
        """
        difference_array = np.abs(array_1 - array_2)
        return np.max(difference_array) < max_delta

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
        shape_mat = np.array([np.reshape(shape.points, (-1)) for shape in shapes])

        pca_obj = PCA()
        pca_obj.fit(shape_mat)

        return cls(mean_shape, pca_obj.components_, pca_obj.explained_variance_)

    @classmethod
    def from_image_shapes(cls, image_shapes: List[ImageShape], des_expvar_ratio=0.95):
        """
        Creates an ActiveShapeModel instance by finding the mean of
        a list of Shapes, then applying PCA to the list to derive the
        parameters with highest variance. Finally, we filter the eigenvalues
        by order of variance until we reach the desired explained
        variance.
        """
        # Get shapes
        shapes = [image_shape.shape for image_shape in image_shapes]
        # Find mean shape
        mean_shape = Shape.mean_from_many(shapes)
        mean_shape.translate_to_origin()
        # Create matrix from all shapes
        shape_mat = np.array([np.reshape(shape.points, (-1)) for shape in shapes])

        pca_obj = PCA()
        pca_obj.fit(shape_mat)

        profiles = GrayLevelProfile.from_image_shapes(image_shapes)

        return cls(
            mean_shape, pca_obj.components_, pca_obj.explained_variance_, profiles
        )

    def propose_shape(self, image_shape: ImageShape):
        """
        Proposes a better shape by sliding each point along its normal axis
        so that the Mahalanobis distance of its profile in relation to the
        mean profile is normalized.
        """
        # unpack
        image = image_shape.image
        shape = image_shape.shape

        # get points and vectors
        points = shape.as_point_list()
        vectors = shape.get_orthogonal_vectors()
        glps = self.gray_level_profiles

        proposed_points = []

        # find m sliding profiles of (2k+1) length for each point
        for point, vector, glp in zip(points, vectors, glps):
            possible_profiles = GrayLevelProfile.sliding_profiles(
                image, point, vector, glp.half_sampling_size
            )
            distances = [
                glp.mahalanobis_distance(profile) for profile in possible_profiles
            ]
            distance_index, _ = min(
                enumerate(np.abs(distances)), key=operator.itemgetter(1)
            )

            proposed_points.append(
                glp.get_point_position(
                    distance_index, len(possible_profiles), point, vector
                )
            )

        return shape(proposed_points)
