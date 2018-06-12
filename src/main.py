"""Main module for Incisor ASM recognition"""

from imgutils import load_images
from image_shape import ImageShape
from shape import Shape
from incisors import Incisors


def main():
    """Main function"""
    images = load_images(range(1, 15))
    active_shape_models, image_shapes = Incisors.active_shape_models(
        images, incisors=[Incisors.UPPER_OUTER_LEFT]
    )

    asm = active_shape_models[Incisors.UPPER_OUTER_LEFT]
    imgshp = image_shapes[Incisors.UPPER_OUTER_LEFT]

    test_imgshp = ImageShape(
        imgshp[0].image, Shape([p.add_noise() for p in imgshp[0].shape.as_point_list()])
    )

    shap = asm.propose_shape(test_imgshp)

    return


if __name__ == "__main__":
    main()
