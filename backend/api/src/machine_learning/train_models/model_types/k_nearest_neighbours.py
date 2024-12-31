from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from ...custom_metric import mydist

PROBLEM_TYPE_TO_CLASS = {
    'classification': KNeighborsClassifier,
    'regression': KNeighborsRegressor
}

def train_knn(train_feature_data_scaled, train_result_data, problem_type, num_continuous_features, num_categorical_features):
    """
    Trains a K-Nearest Neighbours model.

    Parameters
    ----------
    train_feature_data_scaled : pandas.DataFrame
        The scaled feature data to use for training.
    train_result_data : pandas.DataFrame
        The result data to use for training.
    problem_type : str
        The type of problem to train for, either 'classification' or 'regression'.
    num_continuous_features : int
        The number of continuous features in the dataset.
    num_categorical_features : int
        The number of categorical features in the dataset.

    Returns
    -------
    knn_model : sklearn.neighbors.KNeighborsClassifier or sklearn.neighbors.KNeighborsRegressor
        The trained K-Nearest Neighbours model.
    """

    if num_categorical_features == 0:
        metric = 'minkowski'
    elif num_continuous_features == 0:
        metric = 'hamming'
    else:
        metric = mydist
        # Todo: Update the keys in metric_params
        metric_params = {
            'ncts': num_continuous_features,
            'ncate': num_categorical_features
        }
        algorithm = 'ball_tree'
        leaf_size = 2

    knn_class = PROBLEM_TYPE_TO_CLASS[problem_type]
    knn_model = knn_class(
        n_neighbors=min(4, len(train_feature_data_scaled)),
        weights='distance',
        metric=metric,
        metric_params=metric_params,
        algorithm=algorithm,
        leaf_size=leaf_size
    )
    knn_model.fit(train_feature_data_scaled, train_result_data)

    return knn_model
