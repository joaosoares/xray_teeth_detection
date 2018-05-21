import cv2
from shape import Shape
from active_shape_model import ActiveShapeModel
import matplotlib.pyplot as plt


def show_labeler_window(cur_img,
                        landmark_pairs,
                        prev_img=None,
                        prev_img_points=None):
    cv2.namedWindow('Training set labeler', cv2.WINDOW_KEEPRATIO)
    # The first image in our set
    if prev_img is None:
        for pairs in landmark_pairs:
            for pair in pairs:
                cv2.circle(cur_img, pair, 3, (0, 255, 0))
        cv2.imshow('Training set labeler', cur_img)


def get_landmark_pairs(landmarks_filename):
    with open(landmarks_filename) as f:
        points = [int(float(x)) for x in f.readlines()]
        it = iter(points)
        return list(zip(it, it))


def main():
    # base_path = './data/Radiographs/'
    landmarks_path = './data/Landmarks/original/'
    # cur_img = cv2.imread(base_path + '01.tif')
    teeth_points = []
    for i in range(1, 15):
        teeth_points.append(
            get_landmark_pairs(landmarks_path + "landmarks{}-1.txt".format(i)))

    teeth = [Shape(tooth_points) for tooth_points in teeth_points]
    Shape.apply_procrustes(teeth)

    ActiveShapeModel.from_shapes(teeth)

    # for tooth in teeth:
    #     plt.plot(tooth.points[:, 0], tooth.points[:, 1])
    # plt.show()


if __name__ == '__main__':
    main()
