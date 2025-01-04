from .graph_types.one_d_function_plot import generate_1d_function_plot
from .graph_types.two_d_function_plot import generate_2d_function_plot
from .graph_types.two_d_region_plot import generate_2d_region_plot
from .graph_types.tree_graph import generate_tree_graph
import matplotlib.pyplot as plt

def generate_model_graph(
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
):
    """
    Generate a model graph image.

    This function creates a plot of the model. The type of plot is determined by
    the problem type and the number of continuous and categorical features.

    Parameters
    ----------
    model : object
        The trained machine learning model to generate predictions from.
    model_type : str
        The type of model, either 'DT' for decision tree or 'KNN' for k nearest neighbours.
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
    graph_image_base_64 = None

    if (problem_type == 'regression' and len(continuous_features) == 1 and len(categorical_features) == 0):
        # Create a 1d function plot of prediction values
        graph_image_base_64 = generate_1d_function_plot(
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
        )
    elif (problem_type == 'regression' and len(continuous_features) == 2 and len(categorical_features) == 0):
        # Create a 2d surface function plot of prediction values
        graph_image_base_64 = generate_2d_function_plot(
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
        )
    elif (problem_type == 'classification' and len(continuous_features) == 2 and len(categorical_features) == 0):
        # Create 2d graph of coloured regions signifying result values
        graph_image_base_64 = generate_2d_region_plot(
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
        )
    elif (model_type == 'DT'):
        # Create a tree graph
        graph_image_base_64 = generate_tree_graph(
            model,
            model_type,
            problem_type,
            continuous_features,
            categorical_features,
            result_categories_map
        )

    # Clear plot object (for next request)
    plt.clf()

    return graph_image_base_64
