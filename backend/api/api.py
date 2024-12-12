from flask import Flask, render_template, request, send_from_directory
from flask_cors import CORS, cross_origin
import time

import api.src.machineLearning.MachineLearner_Functions as MachineLearner_Functions

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

@app.route('/api/mlFieldIdentifier', methods=['POST'])
@cross_origin()
def mlFieldIdentifier():
    return {'mlDatasetFields': MachineLearner_Functions.fieldIdentifier(request.json["filename"])}

@app.route('/api/mlDatasetSaver', methods=['POST'])
@cross_origin()
def mlDatasetSaver():
    return {'mlDatasetFields': MachineLearner_Functions.datasetSave( request.json["filename"], request.json["dataset"] )}

@app.route('/api/clearModel', methods=['POST'])
@cross_origin()
def mlClearRepresentation():
    return {'mlClearRepresentation': MachineLearner_Functions.clearModel( request.get_json()['sessionid'])}


# Test adding a route
from api.routes.machineLearning import ml_routes
app.register_blueprint(ml_routes, url_prefix='/api/ml')



# Error handling
# @app.errorhandler(404)
# def not_found_error(error):
#     return jsonify({'error': 'Not Found'}), 404

# @app.errorhandler(500)
# def internal_error(error):
#     return jsonify({'error': 'Internal Server Error'}), 500
