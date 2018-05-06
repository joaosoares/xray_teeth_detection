def get_landmark_vector(landmarks_filename):
    with open(landmarks_filename) as f:
        points = [int(float(x)) for x in f.readlines()]
        it = iter(points)
        return list(zip(it, it))


def main():
    base_path = './_Data/Radiographs/'
    landmarks_path = './_Data/Landmarks/original/'
    cur_img = cv2.imread(base_path + '01.tif')
    landmark_pairs = []
    landmark_pairs_array = []
    for i in range(1, 15):
        cur_landmark_pairs = get_landmark_pairs(
            landmarks_path + "landmarks{}-1.txt".format(i))
        landmark_pairs_array.append(list(chain(*cur_landmark_pairs)))
        landmark_pairs_array.append(cur_landmark_pairs)

    show_labeler_window(cur_img, landmark_pairs)
    while(1):
        if cv2.waitKey(20) == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
