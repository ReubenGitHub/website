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
        request.json['supervision'],
        request.json['problemtype'],
        request.json['mlmethod'],
        request.json['polydeg'],
        request.json['ctsparams'],
        request.json['cateparams'],
        request.json['resultparam'],
        request.json['testprop'],
        request.json['datasetname'],
        request.json['sessionId'])
    }

@app.route('/api/uploadDataset', methods=['POST'])
@cross_origin()
def mlDatasetSaver():
    return {
        'datasetFields': MachineLearner_Functions.choose_dataset(
            request.json['sessionId'],
            request.json['useDefaultDataset'],
            request.json.get('dataset', None) # dataset may not exist if useDefaultDataset is true
        )
    }

@app.route('/api/clearSessionData', methods=['POST'])
@cross_origin()
def mlClearRepresentation():
    return {'mlClearRepresentation': MachineLearner_Functions.clear_session_data(request.json['sessionId'])}


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
