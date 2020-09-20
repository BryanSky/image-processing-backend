from app.object_extraction import extract_object_with_position
from app.line_extraction import extract_lines
import json


def build_graph(image_path):
    """
    :param image:
    :return:
    """
    objects = extract_object_with_position(image_path)
    lines = extract_lines(image_path, objects)
    graph = Graph()
    graph.build_from_data(objects, lines)
    json_string = graph.to_json()
    filename = image_path.replace('.png', '.json')
    with open(filename, 'w') as f:
        json.dump(json_string, f)
    return json_string


def visualize_graph(graph):
    """

    :param graph:
    :return:
    """
    # TODO: implement
    return


class Graph(object):

    def __init__(self):
        self.nodes = []
        self.lines = []
        self.node_counter = 0
        self.edge_counter = 0

    def add_node(self, node):
        self.nodes.append(node)

    def add_line(self, line):
        for node in self.nodes:
            if node.has_line_connection(line):
                if line.start is None:
                    line.start = node.id
                elif line.end is None:
                    line.end = node.id
                else:
                    print("WARN: line already has two endings with {} and {} snf would be assigned to {} too"
                          .format(line.start, line.end, node.id))
        self.lines.append(line)

    def build_from_data(self, items, lines):
        for item in items:
            node = Node(self.node_counter, item['label'])
            node.bounding_box = item['box']
            self.node_counter += 1
            self.add_node(node)
        for line in lines:
            edge = Edge(self.edge_counter)
            for x1, y1, x2, y2 in line:
                edge.set_points([x1, y1], [x2, y2])
                self.edge_counter += 1
                self.add_line(edge)

    def to_json(self):
        data = {
            'nodes': [],
            'edges': []
        }
        for node in self.nodes:
            data['nodes'].append(node.build_data_object())
        for line in self.lines:
            data['edges'].append(line.build_data_object())
        return data

static_type_map = {

}


class Node(object):

    def __init__(self, id, type, matching_threshold=10):
        type = (type.split('/')[-1]).split('.')[0]
        self.id = id
        self.type = type
        self.bounding_box = []
        self.data = {}
        self.matching_threshold = matching_threshold

    def set_bounding_box(self, bounding_box):
        self.bounding_box = bounding_box

    def get_bounding_box(self):
        return self.bounding_box

    def get_type(self):
        return self.type

    def set_type(self, type):
        self.type = type

    def build_data_object(self):
        self.data = {
            'data': {
                'id': self.id,
                'label': self.type
            }
        }
        return self.data

    def has_line_connection(self, line):
        x_min, y_min, x_max, y_max = self.bounding_box
        thresholded_bounding_box = [x_min - self.matching_threshold, y_min - self.matching_threshold,
                                    x_max + self.matching_threshold, y_max + self.matching_threshold]
        x_min, y_min, x_max, y_max = thresholded_bounding_box
        x1, y1 = line.start
        x2, y2 = line.end
        return (x_min <= x1 <= x_max and y_min <= y1 <= y_max) or (x_min <= x2 <= x_max and y_min <= y2 <= y_max)


class Edge(object):

    def __init__(self, id, source=None, target=None):
        self.id = id
        self.source = source
        self.target = target
        self.start = None
        self.end = None

    def build_data_object(self):
        obj = {
            'data': {
                'id': self.id,
                'source': self.source,
                'target': self.target
            }
        }
        return obj

    def set_points(self, start, end):
        self.start = start
        self.end = end

