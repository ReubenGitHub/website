from flask import Flask, render_template, request, send_from_directory
from flask_cors import CORS, cross_origin
import time

import os
import sys

#Adds MachineLearner_Functions to Path for access
print("ADDING PYTHON PATHS")
api_dir = os.path.dirname( __file__ )
modules_dir = os.path.join( api_dir, '..', 'src' )
metric_dir = os.path.join( api_dir, '..', 'src/MLer' )
libs_dir = os.path.join( api_dir, '..', '/.heroku/python/lib/python3.9/site-packages')
sys.path.append(modules_dir)
sys.path.append(metric_dir)
sys.path.append(libs_dir)
import MLer.MachineLearner_Functions as MachineLearner_Functions

app = Flask(__name__, static_folder="../build", static_url_path='/')
cors = CORS(app)

@app.errorhandler(404)
def not_found(e):
    return app.send_static_file("index.html")

@app.route('/')
@cross_origin()
def index():
    return app.send_static_file("index.html")

@app.route('/api/time')
def get_current_time():
    return {'time': time.time()}

@app.route('/api/mlModelFit', methods=['POST'])
@cross_origin()
def mlModelFit():
    return {'mlModelOutputs': MachineLearner_Functions.machineLearner(
        request.get_json()['supervision'],
        request.get_json()['problemtype'],
        request.get_json()['mlmethod'],
        request.get_json()['polydeg'],
        request.get_json()['ctsparams'],
        request.get_json()['cateparams'],
        request.get_json()['resultparam'],
        request.get_json()['testprop'],
        request.get_json()['datasetname'],
        request.get_json()['sessionid'])
    }

@app.route('/api/mlModelPredict', methods=['POST'])
@cross_origin()
def mlModelPredict():
    return {'mlModelPrediction': MachineLearner_Functions.modelPrediction(request.get_json()['predictAt'], request.get_json()['sessionid'])}

@app.route('/api/mlFieldIdentifier', methods=['POST'])
@cross_origin()
def mlFieldIdentifier():
    return {'mlDatasetFields': MachineLearner_Functions.fieldIdentifier(request.json["filename"])}

@app.route('/api/mlDatasetSaver', methods=['POST'])
@cross_origin()
def mlDatasetSaver():
    return {'mlDatasetFields': MachineLearner_Functions.datasetSave( request.json["filename"], request.json["dataset"] )}

@app.route('/api/clearRepresentation', methods=['POST'])
@cross_origin()
def mlClearRepresentation():
    return {'mlClearRepresentation': MachineLearner_Functions.clearRepresentation( request.get_json()['sessionid'])}