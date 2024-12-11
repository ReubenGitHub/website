# from flask import Flask, jsonify
# from .routes.machineLearning import ml_routes



# def create_app(app):
#     print("CREATING APPPPPP")
#     # app = Flask(__name__)

#     # Register Blueprints
#     app.register_blueprint(ml_routes, url_prefix='/api/ml')

#     # Error handling
#     @app.errorhandler(404)
#     def not_found_error(error):
#         return jsonify({'error': 'Not Found'}), 404

#     @app.errorhandler(500)
#     def internal_error(error):
#         return jsonify({'error': 'Internal Server Error'}), 500

#     # return app
