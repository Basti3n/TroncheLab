import flask
from flask import request, jsonify

from src.utils.utils import *

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/api/', methods=['GET'])
def home():
    return "<h1>A.py is working</h1>"


@app.route('/api/linear', methods=['POST'])
def linear():
    image = request.files['image']
    return jsonify({'linear': load_linear_model(image)})


@app.route('/api/mlp', methods=['POST'])
def mlp():
    image = request.files['image']
    return jsonify({'mlp': load_mlp_model(image)})


@app.route('/api/cnn', methods=['POST'])
def cnn():
    image = request.files['image']
    return jsonify({'cnn': load_cnn_model(image)})


@app.route('/api/resnet', methods=['POST'])
def resnet():
    image = request.files['image']
    return jsonify({'resnet': load_resnet_model(image)})


@app.route('/api/all', methods=['POST'])
def all():
    image = request.files['image']
    return jsonify({
        'linear': load_linear_model(image),
        'mlp': load_mlp_model(image),
        'cnn': load_cnn_model(image),
        'resnet': load_resnet_model(image)
    })


app.run()

