from flask import Blueprint, request, jsonify
from werkzeug.exceptions import BadRequest
import api.src.machine_learning as machine_learning

ml_routes = Blueprint('ml_routes', __name__)

@ml_routes.route('/predict', methods=['POST'])
def mlModelPredict():
    """
    Predict the outcome of a machine learning model

    Request JSON should contain the following fields:

    - predictAt: the input features to predict
    - sessionId: the session ID for the model

    Returns a JSON object with the prediction result
    """
    try:
        # Parse and validate request JSON
        request_data = request.get_json()
        if not request_data or 'predictAt' not in request_data or 'sessionId' not in request_data:
            raise BadRequest("Missing required fields 'predictAt' or 'sessionId'.")

        # Extract fields
        predict_at = request_data['predictAt']
        session_id = request_data['sessionId']

        # Perform the prediction
        prediction = machine_learning.MachineLearner_Functions.modelPrediction(predict_at, session_id)

        # Return the result
        return jsonify({ 'prediction': prediction }), 200

    except BadRequest as e:
        # Handle missing or invalid input
        return jsonify({'error': str(e)}), 400

    except ValueError as e:
        # Handle custom errors from modelPrediction, e.g., model not found
        return jsonify({'error': str(e)}), 404

    except Exception as e:
        # Handle unexpected errors
        return jsonify({'error': 'An unexpected error occurred.', 'details': str(e)}), 500
