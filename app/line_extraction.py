import cv2
import numpy as np


def extract_lines(img, objects, kernel_size=5):
    """
    :param kernel_size:
    :param img:
    :return:
    """

    # TODO: implement
    return


def extract_lines_by_color(img):
    red_img = img.copy()
    red_img[:, :, 0] = 0
    red_img[:, :, 1] = 0
    cv2.imshow(red_img)
    indices = np.argwhere((red_img != 0).all())
    print(indices)
    lines = apply_hough_trafo(red_img)


def apply_hough_trafo(img):
    kernel_size = 5
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)
    edges = cv2.Canny(blur_gray, 50, 150)

    line_image = np.copy(img) * 0
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=15, minLineLength=50, maxLineGap=20)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
    return lines
