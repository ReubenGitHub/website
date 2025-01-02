from ..encode_image import encode_graph_image
import matplotlib.pyplot as plt
import numpy

def generate_1d_function_plot(
    model,
    problem_type,
    continuous_features,
    categorical_features,
    result,
    train_feature_data,
    test_feature_data,
    train_result_data,
    test_result_data,
    scale
):
    """
    Generate a 1d function plot of prediction values.

    This function creates a plot with the training and testing data as points, and the
    model's predictions as a line.

    Parameters
    ----------
    model : object
        The trained machine learning model to generate predictions from.
    problem_type : str
        The type of problem, either 'classification' or 'regression'.
    continuous_features : list of str
        The names of the continuous feature columns.
    categorical_features : list of str
        The names of the categorical feature columns.
    result : str
        The name of the result column.
    train_feature_data : pandas.DataFrame
        The feature data used for training the model.
    test_feature_data : pandas.DataFrame
        The feature data used for testing the model.
    train_result_data : pandas.Series
        The actual result data corresponding to the training features.
    test_result_data : pandas.Series
        The actual result data corresponding to the testing features.
    scale : object, optional
        The scaler to use for transforming feature values. If not provided, the
        original feature values will be used.

    Returns
    -------
    str
        The base64 encoded image of the plot.
    """
    can_create_1d_function_plot = (
        problem_type == 'regression'
        and len(continuous_features) == 1
        and len(categorical_features) == 0
    )

    if (not can_create_1d_function_plot):
        raise ValueError(
            f'Cannot create 1d function plot for problem type {problem_type} '
            f'with {len(continuous_features)} continuous features and '
            f'{len(categorical_features)} categorical features.'
        )

    # Plot training and testing data
    plt.scatter(train_feature_data, train_result_data, marker='o', color='#03bffe', label='Training Data')
    plt.scatter(test_feature_data, test_result_data, marker='x', color='#ff845b', label='Testing Data')

    # Plot model predictions
    all_feature_values = numpy.concatenate((train_feature_data.iloc[:, 0], test_feature_data.iloc[:, 0]))
    min_feature_value = numpy.floor(min(all_feature_values))
    max_feature_value = numpy.ceil(max(all_feature_values))
    feature_range = numpy.linspace(int(min_feature_value)-1, int(max_feature_value)+1, 200).reshape(-1,1)
    # If scale is provided, transform feature range
    if (scale is not None):
        prediction_points = scale.transform(feature_range)
    else:
        prediction_points = feature_range
    predictions = model.predict(prediction_points)
    plt.plot(feature_range, predictions, color='#fe4203', label='Model')

    # Add axis labels
    plt.xlabel(train_feature_data.columns[0])
    plt.ylabel(result)

    # Add legend
    plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=3)

    # Encode image
    graph_image_base_64 = encode_graph_image('plt', plt, format='png')
    
    return graph_image_base_64
