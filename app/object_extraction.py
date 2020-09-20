import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from app.symbol_detection import load_image, extract_bounding_boxes


def extract_object_with_position(img_path):
    """
    detect and label all objects present in the image
    :param img: 3-channel rgb image
    :return: a list of detected objects; each object should be a seperate class or just a json-Object containing at
    at least the position in the image as well as the label (= class the object belongs to)
    """
    img_gray, img_rgb_detection = load_image(img_path)
    symbols = extract_bounding_boxes(img_gray, img_rgb_detection)
    return symbols


def load_templates():
    files = os.listdir('./../test/symbols_png')
    point_representation_dict = {}
    for file in files:
        point_rep = load_point_representation(os.path.join('./../test/symbols_png', file))
        point_representation_dict[file] = point_rep


def load_point_representation(file):
    print(file)
    img = cv2.imread(file)
    lines = extract_lines_by_color(img)
    points = []
    if lines is None or len(lines) == 0:
        return points
    for line in lines:
        for x1, y1, x2, y2 in line:
            points.append([x1, y1])
            points.append([x2, y2])
    return points


def extract_lines_by_color(img):
    red_img = img.copy()
    red_img[:, :, 0] = 0
    red_img[:, :, 1] = 0
    # cv2.imshow(red_img)
    indices = np.argwhere((red_img != 0).all())
    print(indices)
    lines = apply_hough_trafo(red_img)
    return lines


def apply_hough_trafo(img):
    kernel_size = 5
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)
    edges = cv2.Canny(blur_gray, 50, 150)

    line_image = np.copy(img) * 0
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=5, minLineLength=15, maxLineGap=20)
    if lines is None or len(lines) == 0:
        return lines
    for line in lines:
        for x1, y1, x2, y2 in line:
            # cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
            plt.plot([x1, x2], [y1, y2])
    plt.show()
    return lines

