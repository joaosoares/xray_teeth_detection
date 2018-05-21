import cv2


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
    base_path = './data/Radiographs/'
    landmarks_path = './data/Landmarks/original/'
    cur_img = cv2.imread(base_path + '01.tif')
    landmark_pairs = []
    landmark_pairs_array = []
    for i in range(1, 15):
        cur_landmark_pairs = get_landmark_pairs(landmarks_path +
                                                "landmarks{}-1.txt".format(i))
        landmark_pairs_array.append(list(chain(*cur_landmark_pairs)))
        landmark_pairs_array.append(cur_landmark_pairs)

    show_labeler_window(cur_img, landmark_pairs)
    while (1):
        if cv2.waitKey(20) == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
