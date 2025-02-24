from .session_context_manager import session_context_manager
from .data_handling.preprocess_data import preprocess_data
from .data_handling.split_data import split_data_into_train_and_test
from .data_handling.scale_data import scale_data
from .models.train_model import train_model
from .models.evaluate import calculate_model_metrics
from .models.graphing.generate_image import generate_model_graph
from .models.predict import predict
import matplotlib
matplotlib.use('Agg')

def machineLearner(supervision, problem_type, model_type, poly_degree, continuous_features, categorical_features, result, test_proportion, session_id):
    # Get dataset from session data
    dataset = session_context_manager.get_session_data(session_id)['dataset']
    fields_of_interest = continuous_features + categorical_features + [result]
    dataset = dataset.dropna(axis=0, how='any', thresh=None, subset=fields_of_interest, inplace=False)

    # Preprocess data to split out feature and result data, and handle category encoding (for categorical features or results)
    feature_data, categorical_features_category_maps, result_data, result_categories_map = preprocess_data(
        dataset, categorical_features, continuous_features, result, model_type, problem_type
    )

    # Split data into test and train sets
    train_feature_data, test_feature_data, train_result_data, test_result_data = split_data_into_train_and_test(
        feature_data, result_data, test_proportion
    )

    # Apply scaling to continuous features if required
    train_feature_data_scaled, test_feature_data_scaled, scale = scale_data(
        train_feature_data, test_feature_data, continuous_features, model_type
    )

    # If scaling was applied, convert data to numpy arrays (legacy support)
    # Todo: work out why this was done and if there's a nicer way to handle this
    if (scale is not None):
        train_feature_data_scaled = train_feature_data_scaled.to_numpy()
        test_feature_data_scaled = test_feature_data_scaled.to_numpy()

    # Todo: tidy this up
    if (model_type == 'LinReg' and len(continuous_features) == 0):
        train_feature_data_scaled = train_feature_data_scaled.to_numpy()
        test_feature_data_scaled = test_feature_data_scaled.to_numpy()

    # Train model
    training_parameters = {
        'problem_type': problem_type,
        'num_continuous_features': len(continuous_features),
        'num_categorical_features': len(categorical_features),
        'poly_degree': poly_degree
    }
    model = train_model(train_feature_data_scaled, train_result_data, model_type, training_parameters)
    session_context_manager.add_model(session_id, model)

    # Save model settings
    model_settings = {
        'feature_data_columns': feature_data.columns.tolist(),
        'model_type': model_type,
        'problem_type': problem_type,
        'continuous_features': continuous_features,
        'categorical_features': categorical_features,
        'categorical_features_category_maps': categorical_features_category_maps,
        'result_categories_map': result_categories_map,
        'scale': scale
    }
    session_context_manager.add_model_settings(session_id, model_settings)

    # Calculate model metrics (accuracy etc...)
    model_metrics = calculate_model_metrics(
        model,
        problem_type,
        train_feature_data_scaled,
        test_feature_data_scaled,
        train_result_data,
        test_result_data
    )

    # Generate model graph
    graph_image_base_64 = generate_model_graph(
        model,
        model_type,
        problem_type,
        continuous_features,
        categorical_features,
        result,
        train_feature_data,
        test_feature_data,
        train_result_data,
        test_result_data,
        scale,
        result_categories_map
    )

    # Define input validation object
    allowed_feature_values_for_prediction = [
        list(categorical_features_category_maps[feature].keys())
        for feature in categorical_features
    ]

    # Define outputs
    outputs = {
        'model_metrics': model_metrics,
        'graph_image_base_64': graph_image_base_64,
        'allowed_feature_values_for_prediction': allowed_feature_values_for_prediction
    }

    return outputs

# Evaluate model at some particular value
def modelPrediction(predictAt, session_id):
    session_data = session_context_manager.get_session_data(session_id)
    model = session_data['model']
    model_settings = session_data['model_settings']

    problem_type = model_settings['problem_type']
    model_type = model_settings['model_type']
    continuous_features = model_settings['continuous_features']
    categorical_features = model_settings['categorical_features']
    categorical_features_category_maps = model_settings['categorical_features_category_maps']
    result_categories_map = model_settings['result_categories_map']
    feature_data_columns = model_settings['feature_data_columns']
    scale = model_settings['scale']

    prediction = predict(
        model,
        model_type,
        problem_type,
        scale,
        continuous_features,
        categorical_features,
        categorical_features_category_maps,
        feature_data_columns,
        result_categories_map,
        predictAt # Pre-scaling and encoding
    )

    return { 'predictAt': predictAt, 'prediction': prediction }

# Ensure this is called when the website is closed or page refreshed
def clear_session_data(session_id):
    session_context_manager.remove_session_data(session_id)
    return
