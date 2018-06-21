"""Main module for Incisor ASM recognition"""
import logging
from typing import Callable, Dict, List, Text, Tuple, Type, Union

import click
import numpy as np
import matplotlib.pyplot as plt

from auto_initializator import AutoInitializator
from data_preprocessing import Preprocessor
from evaluator import Evaluator
from imgutils import load_images
from incisors import Incisors
from manual_initializator import Initializator, ManualInitializator

Image = Type[np.ndarray]
FnWithArgs = Tuple[Callable[..., Image], Dict[Text, int]]
FnWithoutArgs = Tuple[Callable[[Image], Image]]
FunctionList = List[Union[FnWithArgs, FnWithoutArgs]]


@click.command()
def preprocess():
    # Load images
    print("Loading images... ", end="")
    images = load_images(range(1, 15))
    print("OK")

    # Preprocess images
    print("Preprocessing images... ", end="")
    preprocessing_pipeline: FunctionList = [
        (Preprocessor.bilateral, {"times": 2}),
        (Preprocessor.sobel, {"scale": 1, "delta": 0}),
    ]
    preprocessed_images = Preprocessor.apply(preprocessing_pipeline, images)

    # Plot subplots
    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.imshow(images[3], cmap="gray")
    ax2.imshow(preprocessed_images[3], cmap="gray")
    plt.show()


@click.command()
@click.option("--auto/--manual")
def evaluate(auto: bool = False):
    """Evaluation function"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    print("Loading images... ", end="")
    # Load images
    images = load_images(range(1, 15))
    print("OK")

    # Preprocess images
    print("Preprocessing images... ", end="")
    preprocessing_pipeline: FunctionList = [
        (Preprocessor.bilateral, {"times": 2}),
        (Preprocessor.sobel, {"scale": 1, "delta": 0}),
    ]
    preprocessed_images = Preprocessor.apply(preprocessing_pipeline, images)

    print("OK")

    # Calculate asms for initialization
    print("Finding initial ASMs... ", end="")
    active_shape_models, expected_image_shapes = Incisors.active_shape_models(
        preprocessed_images
    )
    print("OK")
    # Initialization
    print("Initializing ", end="")
    initializator: Initializator
    if auto:
        print("(auto-initialization)... ", end="")
        initializator = AutoInitializator()
    else:
        print("(manual initialization)... ", end="")
        initializator = ManualInitializator()
    initial_image_shapes = initializator.initialize(
        active_shape_models, preprocessed_images
    )
    print("OK")

    # Create image shapes with normal images for saving
    _, printable_image_shapes = Incisors.active_shape_models(images)

    # Evaluation
    print("Evaluating...")
    evaluator = Evaluator(
        initial_image_shapes, expected_image_shapes, printable_image_shapes
    )
    mean_squared_errors = evaluator.quantitative_eval()
    print(mean_squared_errors)
    evaluator.qualitative_eval()

    return


if __name__ == "__main__":
    evaluate()
