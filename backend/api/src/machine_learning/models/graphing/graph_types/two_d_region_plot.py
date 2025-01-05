from ..encode_image import encode_graph_image
from ..graph_utils import generate_feature_arrays, generate_feature_grid
import matplotlib.pyplot as plt
import numpy
import colorsys
from matplotlib import cm

def generate_2d_region_plot(
    model,
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
):
    """
    Generate a 2D region plot of prediction results.

    This function creates a plot with the training and testing data as points, and the
    model's predictions as colored regions representing different categories.

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
    result_categories_map : dict
        A mapping from result categories to the encoded values.

    Returns
    -------
    str
        The base64 encoded image of the plot.
    """
    can_create_2d_region_plot = (
        problem_type == 'classification'
        and len(continuous_features) == 2
        and len(categorical_features) == 0
    )

    if (not can_create_2d_region_plot):
        raise ValueError(
            f'Cannot create 2d region plot for problem type {problem_type} '
            f'with {len(continuous_features)} continuous features and '
            f'{len(categorical_features)} categorical features.'
        )

    # Define colormaps
    num_result_categories = len(result_categories_map)
    darkcmap = cm.get_cmap('rainbow', num_result_categories)
    lightcmap = cm.get_cmap('rainbow', num_result_categories)
    lightcolours=[]
    for index in range(num_result_categories):
        c = colorsys.rgb_to_hls(lightcmap(index)[0], lightcmap(index)[1], lightcmap(index)[2])
        lightcolours.append( tuple(colorsys.hls_to_rgb(c[0], 1 - 0.5 * (1 - c[1]), c[2]*0.7)) )
    lightcmap = cm.colors.ListedColormap(lightcolours)

    # Plot model predictions
    feature_grids_by_feature, feature_grid = generate_feature_grid(
        train_feature_data,
        test_feature_data,
        continuous_features
    )
    prediction_points = scale.transform(feature_grid) if scale is not None else feature_grid
    predictions = model.predict(prediction_points).reshape(feature_grids_by_feature[continuous_features[0]].shape)
    result_encoded_values = result_categories_map.values()
    plt.pcolormesh(
        *feature_grids_by_feature.values(),
        predictions,
        cmap=lightcmap,
        vmin=min(result_encoded_values),
        vmax=max(result_encoded_values)
    )

    # Plot training and testing data
    result_count = 0
    for result in result_categories_map:
        # Display at most 18 results in legend
        result_count += 1
        hide_result = (result_count>18)

        result_encoded_value = result_categories_map[result]
        train_indices_with_result = train_result_data[train_result_data == result_encoded_value].index
        plt.scatter(
            train_feature_data[continuous_features[0]][train_indices_with_result],
            train_feature_data[continuous_features[1]][train_indices_with_result],
            color=darkcmap(result_encoded_value),
            marker='o',
            label=hide_result*"_"+result[0:8]
        )
        test_indices_with_result = test_result_data[test_result_data == result_encoded_value].index
        plt.scatter(
            test_feature_data[continuous_features[0]][test_indices_with_result],
            test_feature_data[continuous_features[1]][test_indices_with_result],
            color=darkcmap(result_encoded_value),
            marker='x'
        )
    
    # Add axis labels
    plt.xlabel(train_feature_data.columns[0])
    plt.ylabel(train_feature_data.columns[1])

    # Add legend
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")

    # Encode image
    graph_image_base_64 = encode_graph_image('plt', plt, format='png')

    return graph_image_base_64
