import numpy as np
import cv2

templates = [
    ('symbols_cutted/Cns.png', 0.7),
    ('symbols_cutted/Gen.png', 0.9),
    ('symbols_cutted/Mxv-1.png', 0.9),
    ('symbols_cutted/Mxv-2.png', 0.9),
    ('symbols_cutted/Hyds.png', 0.9),
    ('symbols_cutted/HExg-1.png', 0.85),
    ('symbols_cutted/Stk.png', 0.85),
    ('symbols_cutted/Pu.png', 0.9),
    ('symbols_cutted/Pu_flipped.png', 0.9),
    ('symbols_cutted/TSen.png', 0.8),
    ('symbols_cutted/VflSen.png', 0.9)
]


def load_image(img_path):
    # img_path = "examples_png/ex1.png"
    # img_path = "examples_png/ex2a.png"
    # img_path = "examples_png/ex2b.png"

    # Load an color image as it is and in grayscale
    img_rgb = cv2.imread(img_path, flags=1)  # height, width
    img_gray = cv2.imread(img_path, flags=0)

    width = 1000
    scale = img_rgb.shape[0] / img_rgb.shape[1]
    assert scale <= 1.1, "FabError: Image has wrong format!"
    height = round(width * scale)
    img_rgb = cv2.resize(img_rgb, dsize=(width, height), interpolation=cv2.INTER_AREA)
    img_gray = cv2.resize(img_gray, dsize=(width, height), interpolation=cv2.INTER_AREA)
    img_rgb_detection = img_rgb.copy()
    return img_gray, img_rgb_detection


###############################
###### Template Matching ######
###############################

# TODO use center of the bounding boxes for distance calculation using width and height
def non_maximum_suppression(loc, scores, threshold=250, widths=[], heights=[]):
    assert len(loc[0]) == len(loc[1]) == len(scores), "FabError: loc should be a list of lists where the sublists have the same length!"
    if len(widths) > 0:
        assert len(loc[0]) == len(widths) == len(heights), "FabError: These lists should have the same length!"

    # loc = list(loc_new)
    # loc_reduced = []
    i = 0
    while i < len(loc[0]):
        j = i + 1
        x1, y1 = loc[0][i], loc[1][i]
        while j < len(loc[0]):
            x2, y2 = loc[0][j], loc[1][j]
            if ((x1-x2)**2 + (y1-y2)**2) < threshold:
                if scores[i] > scores[j]:
                    del loc[0][j]
                    del loc[1][j]
                    del scores[j]
                    if len(widths) > 0:
                        del widths[j]
                        del heights[j]
                else:
                    del loc[0][i]
                    del loc[1][i]
                    del scores[i]
                    if len(widths) > 0:
                        del widths[i]
                        del heights[i]
                    i -= 1
                    break
            else:
                j += 1
        i += 1

    if len(widths) > 0:
        return loc, scores, widths, heights
    return loc, scores


def extract_bounding_boxes(img_gray, img_rgb_detection):
    symbols = []
    colors = iter([(0, 0, 255), (0, 127, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 0, 127),
                   (255, 0, 255), (127, 0, 255), (0, 255, 127), (127, 255, 0), (255, 255, 0), (255, 127, 0)
                   ] + 20 * [(127, 0, 255)])
    for template_path, threshold in templates:
        color = next(colors)
        img_template = cv2.imread(template_path, 0)
        # loop over the scales of the template image

        loc = [[], []]
        score_reduced = []
        widths, heights = [], []

        for scale in np.linspace(0.3, 0.8, 10)[::-1]:

            tmp_resized = cv2.resize(img_template, dsize=(round(img_template.shape[1]*scale), round(img_template.shape[0]*scale)),
                                     interpolation=cv2.INTER_AREA)

            w, h = tmp_resized.shape[::-1]

            # the following two lines do the object recognition based on template matching
            score = cv2.matchTemplate(img_gray, tmp_resized, cv2.TM_CCOEFF_NORMED)
            loc_new = np.where(score >= threshold)

            # use non-maximum-suppression to remove multiple detections of the same object
            if loc_new[0].size > 0:
                scores = list(score[loc_new])
                loc_new = [loc_new[0].tolist(), loc_new[1].tolist()]
                loc_reduced, score_reduced_loc = non_maximum_suppression(loc_new, scores)
                score_reduced.extend(score_reduced_loc)
                loc[0].extend(loc_reduced[0])
                loc[1].extend(loc_reduced[1])
                widths.extend(len(loc_reduced[0]) * [w])
                heights.extend(len(loc_reduced[0]) * [h])
        if len(loc[0]) > 0:
            loc_reduced, score_reduced, widths, heights = non_maximum_suppression(loc, score_reduced, widths=widths, heights=heights)

            # plot bounding boxes
            width_it = iter(widths)
            height_it = iter(heights)
            for pt in zip(*loc[::-1]):
                w, h = next(width_it), next(height_it)
                cv2.rectangle(img_rgb_detection, pt1=pt, pt2=(pt[0] + w, pt[1] + h), color=color, thickness=2)
                pt1_x, pt1_y = pt1_x
                pt2_x = pt[0] + w
                pt2_y = pt[1] + h
                symbols.append([min(pt1_x, pt2_x), min(pt1_y, pt2_y), max(pt1_x, pt2_x), max(pt1_y, pt2_y)])
                print(template_path, len(score_reduced))
    # cv2.imshow('Image with detections', img_rgb_detection)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print("End of circle_detection.py")
    return symbols
