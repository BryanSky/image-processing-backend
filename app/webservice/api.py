from flask import Blueprint, jsonify, url_for, request, send_file
from threading import Thread
import os
import cv2 as cv
import json
import uuid
import sys
import time

from app.graph_builder import build_graph

blueprint = Blueprint('sample', __name__, url_prefix='/')


@blueprint.route('/')
def welcome():
    return 'welcome'


def evaluate_image(img):
    print(img)
    return build_graph(img)


@blueprint.route('/upload', methods=['POST'])
def upload_image():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            generated_id = uuid.uuid1()
            print("OK -1")
            print(image)
            image.save("app/data/" + str(generated_id) + ".png")
            print("OK")
            # thread = Thread(target=evaluate_image, args=("app/data/" + str(generated_id) + ".png", ))
            print("OK 1/2")
            # thread.start()
            print("OK 1")
            try:
                evaluate_image("app/data/" + str(generated_id) + ".png")
            except:
                print("Something went wrong")
            return jsonify(str(generated_id))


@blueprint.route('/preview/<image_id>', methods=['GET'])
def get_image(image_id):
    image = 'data/' + str(image_id)
    return send_file(image, mimetype='image/png')


@blueprint.route('/data/<image_id>', methods=['GET'])
def get_graph(image_id):
    graph_file = 'app/data/' + str(image_id)
    if not os.path.isfile(graph_file):
        print("not found {}".format())
        sys.stdout('Status: 404 Not Found\r\n\r\n')
    with open(graph_file) as json_file:
        graph = json.load(json_file)
    if graph is not None:
        return jsonify(graph)
    else:
        sys.stdout('Status: 404 Not Found\r\n\r\n')
