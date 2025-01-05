import numpy
import pandas

def generate_feature_arrays(feature_data, features):
    """
    Extract arrays of feature values from the feature data.

    This function takes a dataset and a list of feature names, and returns a list
    of numpy arrays corresponding to the values of each specified feature.

    Parameters
    ----------
    feature_data : pandas.DataFrame
        The dataset containing feature columns.
    features : list of str
        The names of the features to extract.

    Returns
    -------
    list of numpy.ndarray
        A list containing the arrays of values for each specified feature.
    """
    return {feature: feature_data[feature].values for feature in features}

def generate_feature_grid(
    train_feature_data,
    test_feature_data,
    continuous_features
):
    """
    Generate a grid of feature values for a model.

    This function generates a grid of all possible combinations of feature values within a
    given range. The range is determined by the minimum and maximum values of the feature
    in the training and testing datasets.

    Parameters
    ----------
    train_feature_data : pandas.DataFrame
        The feature data used for training the model.
    test_feature_data : pandas.DataFrame
        The feature data used for testing the model.
    continuous_features : list of str
        The names of the continuous feature columns.

    Returns
    -------
    feature_grids_by_feature : dict
        A dictionary mapping each feature to its grid of values.
    feature_grid : numpy.array
        The grid of feature values, where each row is a set of values for all features.
    """
    feature_data = pandas.concat([train_feature_data, test_feature_data])

    feature_ranges_by_feature = {}
    for feature in continuous_features:
        feature_min_value = numpy.floor(min(feature_data[feature]))
        feature_max_value = numpy.ceil(max(feature_data[feature]))
        feature_ranges_by_feature[feature] = numpy.linspace(
            int(feature_min_value)-1,
            int(feature_max_value)+1,
            200
        )

    feature_grids_by_feature = {
        feature: grid
        for feature, grid in zip(continuous_features, numpy.meshgrid(*feature_ranges_by_feature.values()))
    }

    feature_grid = numpy.vstack([grid.ravel() for grid in feature_grids_by_feature.values()]).transpose()

    return feature_grids_by_feature, feature_grid
