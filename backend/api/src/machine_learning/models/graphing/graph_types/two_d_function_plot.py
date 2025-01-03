from ..encode_image import encode_graph_image
import matplotlib.pyplot as plt
import numpy

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
    train_feature_1_data = train_feature_data.iloc[:, 0]
    train_feature_2_data = train_feature_data.iloc[:, 1]
    axes.scatter(
        train_feature_1_data,
        train_feature_2_data,
        train_result_data,
        marker='o',
        color='#03bffe',
        label='Training Data'
    )
    test_feature_1_data = test_feature_data.iloc[:, 0]
    test_feature_2_data = test_feature_data.iloc[:, 1]
    axes.scatter(
        test_feature_1_data,
        test_feature_2_data,
        test_result_data,
        marker='x',
        color='#ff845b',
        label='Testing Data'
    )

    # Plot model predictions
    all_feature_1_values = numpy.concatenate((train_feature_data.iloc[:, 0], test_feature_data.iloc[:, 0]))
    all_feature_2_values = numpy.concatenate((train_feature_data.iloc[:, 1], test_feature_data.iloc[:, 1]))
    min_feature_1_value = numpy.floor(min(all_feature_1_values))
    max_feature_1_value = numpy.ceil(max(all_feature_1_values))
    min_feature_2_value = numpy.floor(min(all_feature_2_values))
    max_feature_2_value = numpy.ceil(max(all_feature_2_values))
    feature_1_range = numpy.linspace(int(min_feature_1_value)-1,int(max_feature_1_value)+1,200)
    feature_2_range = numpy.linspace(int(min_feature_2_value)-1,int(max_feature_2_value)+1,200)
    feature_1_grid, feature_2_grid = numpy.meshgrid( feature_1_range, feature_2_range )
    feature_grid = numpy.vstack([feature_1_grid.ravel(), feature_2_grid.ravel()]).transpose()
    # If scale is provided, transform feature grid
    if (scale is not None):
        prediction_points = scale.transform(feature_grid)
    else:
        prediction_points = feature_grid
    predictions = model.predict(prediction_points).reshape(feature_1_grid.shape)
    surface = axes.plot_surface(
        feature_1_grid,
        feature_2_grid,
        predictions,
        alpha = 0.5,
        cmap='copper',
        color="#03bffe",
        label='Model'
    )
    surface._edgecolors2d = surface._edgecolor3d
    surface._facecolors2d = surface._facecolor3d

    # Add axis labels
    axes.set_xlabel(train_feature_data.columns[0])
    axes.set_ylabel(train_feature_data.columns[1])
    axes.set_zlabel(result)

    # Add legend
    axes.legend(bbox_to_anchor=(0.5,1.1), loc="upper center", ncol=3)

    # Encode image
    graph_image_base_64 = encode_graph_image('plt', plt, format='png')
    
    return graph_image_base_64
