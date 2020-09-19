import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering


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
    for line in lines:
        for x1, y1, x2, y2 in line:
            # cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
            plt.plot([x1, x2], [y1, y2])
    plt.show()
    return lines


def cluster_lines(lines, threshold=20):
    clusters = {}
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


def apply_agglomerative_clustering(lines):
    points = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            points.append([x1, y1])
            points.append([x2, y2])
    # dendrogram = sch.dendrogram(sch.linkage(points, method='ward'))
    hc = AgglomerativeClustering(n_clusters=13, affinity='euclidean', linkage='ward')
    y_hc = hc.fit_predict(points)
    y_hc = np.array(y_hc)
    points = np.array(points)
    plt.show()
    print(y_hc)
    plt.scatter(points[y_hc == 0, 0], points[y_hc == 0, 1], s=100, c=(random_color()/255))
    plt.scatter(points[y_hc == 1, 0], points[y_hc == 1, 1], s=100, c=(random_color()/255))
    plt.scatter(points[y_hc == 2, 0], points[y_hc == 2, 1], s=100, c=(random_color()/255))
    plt.scatter(points[y_hc == 3, 0], points[y_hc == 3, 1], s=100, c=(random_color()/255))

    plt.scatter(points[y_hc == 4, 0], points[y_hc == 4, 1], s=100, c=(random_color()/255))
    plt.scatter(points[y_hc == 5, 0], points[y_hc == 5, 1], s=100, c=(random_color()/255))
    plt.scatter(points[y_hc == 6, 0], points[y_hc == 6, 1], s=100, c=(random_color()/255))
    plt.scatter(points[y_hc == 7, 0], points[y_hc == 7, 1], s=100, c=(random_color()/255))

    plt.scatter(points[y_hc == 8, 0], points[y_hc == 8, 1], s=100, c=(random_color()/255))
    plt.scatter(points[y_hc == 9, 0], points[y_hc == 9, 1], s=100, c=(random_color()/255))
    plt.scatter(points[y_hc == 10, 0], points[y_hc == 10, 1], s=100, c=(random_color()/255))
    plt.scatter(points[y_hc == 11, 0], points[y_hc == 11, 1], s=100, c=(random_color()/255))

    plt.scatter(points[y_hc == 12, 0], points[y_hc == 12, 1], s=100, c=(random_color()/255))
    plt.show()


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


def test():
    img_path = "./../test/ex2a.png"
    img = cv2.imread(img_path)
    lines = extract_lines_by_color(img)
    # apply_agglomerative_clustering(lines)
    clusters = cluster_lines(lines)
    plot_clusters(clusters)
