import flask
from os import path
from flask import request, jsonify
from flask_cors import CORS, cross_origin

from src.utils.utils import *

app = flask.Flask(__name__)
app.config["DEBUG"] = True
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


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


@app.route('/api/custom', methods=['POST'])
def custom():
    image = request.files['image']
    pathInput = request.args.get('path') + ".keras"
    print(pathInput)
    if not path.exists(f'./models/{pathInput}'):
        return 'No file at this path', 400
    return jsonify({'model': load_custom_model(image, pathInput)})


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

