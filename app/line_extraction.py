import cv2
import numpy as np
import random

import matplotlib.pyplot as plt

# ==============this code added==================================================================:
# import pydevd_pycharm

# pydevd_pycharm.settrace('192.168.0.4', port=12345, stdoutToServer=True,
#                         stderrToServer=True)


# ================================================================================================


def extract_lines(img_path, objects, kernel_size=5):
    """
    :param kernel_size:
    :param img:
    :return:
    """
    # img_path = "./../test/hydr0_hand.png"
    img = cv2.imread(img_path)
    img = flatten_colors(img)
    lines = apply_hough_trafo(img)
    # todo: implement
    #  merge_lines()
    clusters = cluster_lines(lines)
    # plot_clusters(clusters)
    return lines


def extract_components(img):
    random.seed(12345)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)
    # Draw contours + hull results
    drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
        cv2.drawContours(drawing, contours, i, color)
        cv2.drawContours(drawing, hull_list, i, color)
    # Show in a window
    cv2.imwrite("./../test/result.png", drawing)


def apply_hough_trafo(img):
    kernel_size = 5
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)
    edges = cv2.Canny(blur_gray, 50, 150)

    line_image = np.copy(img) * 0
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=5, minLineLength=25, maxLineGap=30)
    for line in lines:
        for x1, y1, x2, y2 in line:
            # cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
            plt.plot([x1, x2], [y1, y2])
    plt.show()
    return lines


def cluster_lines(lines, threshold=20):
    clusters = {}
    if lines is None:
        return clusters
    for line in lines:
        for x1, y1, x2, y2 in line:
            p1 = np.array([x1, y1])
            p2 = np.array([x2, y2])
            added = False
            for cluster_num in clusters.keys():
                if len(np.where((clusters[cluster_num] - p1 < threshold).all())) > 0 or len(
                        np.where((clusters[cluster_num] - p2 < threshold).all())) > 0:
                    cluster = clusters[cluster_num]
                    cluster += p1
                    cluster += p2
                    added = True
            if not added:
                cluster = [p1, p2]
                clusters[len(clusters)] = cluster
    return clusters


def random_color():
    rgbl=[255, 0, 0]
    random.shuffle(rgbl)
    return np.array(tuple(rgbl))


def plot_clusters(clusters):
    for key in clusters.keys():
        cluster = clusters[key]
        for i in range(0, len(cluster), 4):
            line = [cluster[i: i+4]]
            if len(line) < 4:
                continue
            for x1, y1, x2, y2 in line:
                plt.plot([x1, x2], [y1, y2])
        plt.show()


def flatten_colors(img):
    bins = 8
    hist_r = cv2.calcHist([img], [0], None, [bins], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [bins], [0, 256])
    hist_b = cv2.calcHist([img], [2], None, [bins], [0, 256])

    hist_sum_sm = np.sum([hist_r[:, 0], hist_g[:, 0], hist_b[:, 0]], axis=0)

    min_vals = np.r_[True, hist_sum_sm[1:] < hist_sum_sm[:-1]] & np.r_[hist_sum_sm[:-1] < hist_sum_sm[1:], True]
    max_vals = np.r_[True, hist_sum_sm[1:] > hist_sum_sm[:-1]] & np.r_[hist_sum_sm[:-1] > hist_sum_sm[1:], True]

    start = 0
    val = 0
    multiplicator = (256 / bins)
    for i in range(bins):
        if min_vals[i]:
            img[(img > start * multiplicator - 1) & (img < i * multiplicator)] = val * multiplicator
            start = i
        elif max_vals[i]:
            val = i
    img[(img > start * multiplicator - 1) & (img < 256)] = val * multiplicator

    # r_channel = img[:, :, 0]
    # g_channel = img[:, :, 1]
    # b_channel = img[:, :, 2]

    # img[np.where((r_channel == g_channel) & (g_channel == b_channel))] = 0
    # kernel = np.ones((5, 5), np.uint8)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # img = cv2.dilate(img, kernel, iterations=1)
    #
    # plt.imshow(img)
    # plt.show()

    # eliminate color channels which have no effect on the lines
    img[:, :, 1] = 0
    img[:, :, 2] = 0
    img[np.where(img[:, :, 0] > 200)] = 0
    img[np.where(img[:, :, 0] < 50)] = 0
    img[np.where(img[:, :, 0] != 0)] = 255

    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    # plt.imshow(erosion)
    # plt.show()
    return erosion
