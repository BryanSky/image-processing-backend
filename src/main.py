import cv2
from .object_extraction import extract_object_with_position
from .line_extraction import extract_lines
from .graph_builder import build_graph, visualize_graph


def process_image(image_path='./../res/example_img.png'):
    """
    basic function to run the image through whole processing chain
    :return: visualization of the graph, built from the input image, in form of an image
    """
    img = cv2.imread(image_path)
    detected_objects = extract_object_with_position(img)
    detected_lines = extract_lines(img, detected_objects)
    graph = build_graph(detected_objects, detected_lines)
    visualized_graph = visualize_graph(graph)
    cv2.imwrite('./../res/example_graph.png')
    return visualized_graph


process_image()
