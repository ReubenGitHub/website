from ..encode_image import encode_graph_image
from ..graph_utils import generate_feature_arrays, generate_feature_grid
import matplotlib.pyplot as plt

def generate_2d_function_plot(
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
    Generate a 2d function plot of prediction values.

    This function creates a plot with the training and testing data as points, and the
    model's predictions as a surface.

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
    can_create_2d_function_plot = (
        problem_type == 'regression'
        and len(continuous_features) == 2
        and len(categorical_features) == 0
    )

    if (not can_create_2d_function_plot):
        raise ValueError(
            f'Cannot create 2d function plot for problem type {problem_type} '
            f'with {len(continuous_features)} continuous features and '
            f'{len(categorical_features)} categorical features.'
        )

    # Instantiate figure
    figure = plt.figure()
    axes = figure.add_subplot(projection='3d')

    # Plot training and testing data
    train_feature_arrays_by_feature = generate_feature_arrays(train_feature_data, continuous_features)
    axes.scatter(
        *train_feature_arrays_by_feature.values(),
        train_result_data,
        marker='o',
        color='#03bffe',
        label='Training Data'
    )
    test_feature_arrays_by_feature = generate_feature_arrays(test_feature_data, continuous_features)
    axes.scatter(
        *test_feature_arrays_by_feature.values(),
        test_result_data,
        marker='x',
        color='#ff845b',
        label='Testing Data'
    )

    # Plot model predictions
    feature_grids_by_feature, feature_grid = generate_feature_grid(
        train_feature_data,
        test_feature_data,
        continuous_features
    )
    prediction_points = scale.transform(feature_grid) if scale is not None else feature_grid
    predictions = model.predict(prediction_points).reshape(feature_grids_by_feature[continuous_features[0]].shape)
    surface = axes.plot_surface(
        *feature_grids_by_feature.values(),
        predictions,
        alpha = 0.5,
        cmap='copper',
        color="#03bffe",
        label='Model'
    )
    surface._edgecolors2d = surface._edgecolor3d
    surface._facecolors2d = surface._facecolor3d

    # Add axis labels
    axes.set_xlabel(continuous_features[0])
    axes.set_ylabel(continuous_features[1])
    axes.set_zlabel(result)

    # Add legend
    axes.legend(bbox_to_anchor=(0.5,1.1), loc="upper center", ncol=3)

    # Encode image
    graph_image_base_64 = encode_graph_image('plt', plt, format='png')
    
    return graph_image_base_64
