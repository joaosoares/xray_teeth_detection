import cv2
from matplotlib import pyplot as plt


class Preprocessor:
    @classmethod
    def bilateral_filter(
        cls, image, diameter=9, sigma_color=150, sigma_space=150, times=1
    ):
        filtered = image
        for _ in range(times):
            filtered = cv2.bilateralFilter(
                image, d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space
            )
        return filtered

    @classmethod
    def median_filter(cls, img, ksize, times):
        filtered = img
        for i in range(times):
            filtered = cv2.medianBlur(img, ksize=ksize)
        return filtered

    @classmethod
    def errosion(cls, img, ksize):
        return cv2.erode(img, kernel=ksize)

    @classmethod
    def dilatation(cls, img, ksize):
        return cv2.dilate(img, kernel=ksize)

    @classmethod
    def top_hat_processing(cls, img, ksize):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(ksize, ksize))
        return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel=kernel)

    @classmethod
    def laplacian(cls, img):
        return cv2.Laplacian(img, ddepth=cv2.CV_64F)

    @classmethod
    def show_image(cls, img):
        plt.imshow(img, cmap="gray")
        plt.show()

    @classmethod
    def apply_sobel(cls, img, scale, delta):
        ddepth = cv2.CV_16S

        grad_x = cv2.Sobel(
            img,
            ddepth,
            1,
            0,
            ksize=3,
            scale=scale,
            delta=delta,
            borderType=cv2.BORDER_DEFAULT,
        )
        grad_y = cv2.Sobel(
            img,
            ddepth,
            0,
            1,
            ksize=3,
            scale=scale,
            delta=delta,
            borderType=cv2.BORDER_DEFAULT,
        )

        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    @classmethod
    def apply_scharr(cls, img, scale, delta):
        ddepth = cv2.CV_16S
        grad_x = cv2.Scharr(
            img, ddepth, 1, 0, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT
        )
        grad_y = cv2.Scharr(
            img, ddepth, 0, 1, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT
        )

        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


if __name__ == "__main__":

    img = cv2.imread("../data/Radiographs/01.tif", flags=cv2.IMREAD_GRAYSCALE)
    img = Preprocessor.bilateral_filter(
        img, diameter=9, sigma_color=150, sigma_space=150, times=1
    )
    # img = Preprocessor.median_filter(img, ksize=5, times=5)
    # img = Preprocessor.top_hat_processing(img, ksize=150)
    img = Preprocessor.apply_sobel(img, scale=1, delta=0)

    Preprocessor.show_image(img)
