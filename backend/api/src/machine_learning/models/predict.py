from ..data_handling.one_hot_encoding import should_use_one_hot_encoding
import numpy

def predict(
    model,
    model_type,
    problem_type,
    scale,
    continuous_features,
    categorical_features,
    categorical_features_category_maps,
    feature_data_columns,
    result_categories_map,
    prediction_feature_values # Pre-scaling and encoding
):
    number_continuous_features = len(continuous_features)
    number_categorical_features = len(categorical_features)

    prediction_point = []

    if number_continuous_features > 0:
        prediction_point_continuous_feature_values = [prediction_feature_values[feature] for feature in continuous_features]
        if scale:
            prediction_point_continuous_feature_values_scaled = scale.transform([prediction_point_continuous_feature_values])[0]
            prediction_point.extend(prediction_point_continuous_feature_values_scaled)
        else:
            prediction_point.extend(prediction_point_continuous_feature_values)

    # Encode categorical features
    if number_categorical_features > 0:
        use_one_hot_encoding = should_use_one_hot_encoding(model_type)
        if use_one_hot_encoding:
            feature_dummy_columns = feature_data_columns[number_continuous_features:]
            number_dummy_columns = len(feature_dummy_columns)
            prediction_categorical_feature_values_encoded = [0] * (number_dummy_columns)
            for feature, value in prediction_feature_values.items():
                if feature in categorical_features:
                    dummy_column_name = feature + "_" + value
                    if dummy_column_name in feature_dummy_columns:
                        dummy_variable_one_position = feature_dummy_columns.index(dummy_column_name)
                        prediction_categorical_feature_values_encoded[dummy_variable_one_position] = 1
        else:
            prediction_categorical_feature_values_encoded = [
                categorical_features_category_maps[feature][prediction_feature_values[feature]]
                for feature in categorical_features
            ]
        
        prediction_point.extend(prediction_categorical_feature_values_encoded)

    # Make prediction
    prediction_point = numpy.array([prediction_point])
    prediction = model.predict(prediction_point)[0]

    if problem_type == 'classification':
        prediction = list(result_categories_map.keys())[prediction]

    return prediction
