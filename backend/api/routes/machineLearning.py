from api.src.machineLearning import MachineLearner_Functions

# @ml_routes.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     prediction = model_prediction(data)
#     return {'prediction': prediction}



from flask import Blueprint, request, jsonify
from werkzeug.exceptions import BadRequest

ml_routes = Blueprint('ml_routes', __name__)

@ml_routes.route('/predict', methods=['POST'])
def mlModelPredict():
    try:
        # Parse and validate request JSON
        request_data = request.get_json()
        if not request_data or 'predictAt' not in request_data or 'sessionid' not in request_data:
            raise BadRequest("Missing required fields 'predictAt' or 'sessionid'.")

        # Extract fields
        predict_at = request_data['predictAt']
        session_id = request_data['sessionid']

        # Perform the prediction
        prediction_result = MachineLearner_Functions.modelPrediction(predict_at, session_id)

        # Return the result
        return jsonify({'mlModelPrediction': prediction_result}), 200

    except BadRequest as e:
        # Handle missing or invalid input
        return jsonify({'error': str(e)}), 400

    except ValueError as e:
        # Handle custom errors from modelPrediction, e.g., model not found
        return jsonify({'error': str(e)}), 404

    except Exception as e:
        # Handle unexpected errors
        return jsonify({'error': 'An unexpected error occurred.', 'details': str(e)}), 500