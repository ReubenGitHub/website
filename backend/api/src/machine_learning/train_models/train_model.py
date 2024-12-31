from .model_types.decision_tree import train_decision_tree
from .model_types.k_nearest_neighbours import train_knn
from .model_types.linear_regression import train_linear_regression
from .model_types.polynomial_regression import train_polynomial_regression

def train_model(train_feature_data_scaled, train_result_data, model_type, training_parameters={}):
    """
    Train a model based on the given scaled feature data, result data, model type and training parameters.

    Parameters
    ----------
    train_feature_data_scaled : pandas.DataFrame
        The scaled feature data to use for training.
    train_result_data : pandas.DataFrame
        The result data to use for training.
    model_type : str
        The type of model to train. Can be 'DT', 'KNN', 'LinReg', 'PolyFit'.
    training_parameters : machine_learning.TrainingParameters
        The parameters to use for training the model.

    Returns
    -------
    model : object
        The trained model.
    """
    if model_type == 'DT':
        model = train_decision_tree(
            train_feature_data_scaled,
            train_result_data,
            training_parameters['problem_type']
        )
    elif model_type == 'KNN':
        model = train_knn(
            train_feature_data_scaled,
            train_result_data,
            training_parameters['problem_type'],
            training_parameters['num_continuous_features'],
            training_parameters['num_categorical_features']
        )
    elif model_type == 'LinReg':
        model = train_linear_regression(
            train_feature_data_scaled,
            train_result_data
        )
    elif model_type == 'PolyFit':
        model = train_polynomial_regression(
            train_feature_data_scaled,
            train_result_data,
            training_parameters['poly_degree']
        )
    else:
        raise ValueError(f'Unsupported model type: {model_type}')

    return model
