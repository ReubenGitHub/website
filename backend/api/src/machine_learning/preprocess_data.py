from ordered_set import OrderedSet
import pandas

def preprocess_data(dataset, categorical_features, continuous_features, result, model_type, problem_type):
    """
    Preprocesses the dataset by separating and encoding features and result data.

    This function preprocesses both feature and result data of a dataset. It handles
    the encoding of categorical features, and processes the result data based on the
    problem type. Categorical features are mapped to integers and, and results
    are mapped to integers for classification problems.

    Args:
        dataset (pandas.DataFrame): The dataset to preprocess.
        categorical_features (list of str): The names of the categorical features.
        continuous_features (list of str): The names of the continuous features.
        result (pandas.Series): The result data to preprocess.
        model_type (str): The type of model to use (e.g. decision tree, linear regression, etc.).
        problem_type (str): The type of problem (e.g. classification, regression).

    Returns:
        feature_data (pandas.DataFrame): The preprocessed feature data.
        categorical_features_category_maps (dict): Maps for categorical feature categories to integers.
        result_data (pandas.Series): The preprocessed result data.
        result_categories_map (dict): A map from result categories to integers, if applicable.

    """
    # Used to have no .copy(), maybe required.
    # Todo: Write tests for testing and training.
    feature_data = dataset[categorical_features + continuous_features].copy()
    result_data = dataset[result].copy()

    feature_data, categorical_features_category_maps = preprocess_feature_data(
        feature_data, categorical_features, continuous_features, model_type
    )

    result_data, result_categories_map = preprocess_result_data(result_data, problem_type)

    return feature_data, categorical_features_category_maps, result_data, result_categories_map

def preprocess_feature_data(feature_data, categorical_features, continuous_features, model_type):
    """
    Preprocess the features of a dataset by encoding categorical features.

    Args:
        feature_data (pandas.DataFrame): The feature data to preprocess.
        categorical_features (list of str): Names of categorical feature columns.
        continuous_features (list of str): Names of continuous feature columns.
        model_type (str): Model type to determine encoding strategy (e.g., 'decision tree',
            'linear regression').

    Returns:
        feature_data (pandas.DataFrame): Processed feature data with categorical features encoded.
        categorical_features_category_maps (dict): Mapping of original categories to integers for
            each categorical feature.
    """
    # Convert categorical features to strings, and the rest to the existing type
    categorical_feature_data_types = {feature: 'str' for feature in categorical_features}
    feature_data = feature_data.astype(dtype=categorical_feature_data_types, copy=True)

    # Create maps from categorical features' categories to integers, and convert categorical feature options to integers
    use_one_hot_encoding = should_use_one_hot_encoding(model_type)
    categorical_features_category_maps = {}
    for feature in categorical_features:
        # Also used for input validation during prediction
        categorical_features_category_maps[feature] = {option: value for value, option in enumerate(OrderedSet(feature_data[feature]))}
        if use_one_hot_encoding:
            # get_dummies targets and replaces the specified column with dummy variables
            feature_data = pandas.get_dummies(data=feature_data, drop_first=True, columns = [feature])
        else:
            feature_data[feature] = feature_data[feature].map(categorical_features_category_maps[feature])
    
    return feature_data, categorical_features_category_maps

def should_use_one_hot_encoding(model_type):
    """
    Whether to convert categorical features to indicator dummy variables, which is required by some models.
    Linear Regression requires this if categories are not ordinal (and we don't separately handle ordinal categories).
    Categories like 'cold', 'medium', 'hot' are ordinal and could be converted to integers like 0, 1, 2

    Args:
        model_type (str): A string indicating the type of model to use. Can be 'DT', 'KNN', 'LinReg', etc.

    Returns:
        bool: Whether to use one hot encoding for categorical features.
    """
    if model_type in ['DT', 'KNN']:
        # Convert each option to an integer
        use_one_hot_encoding = False
    else:
        # Convert each option to a column, which is an indicator dummy variable. A 1 indicates the option is true, a 0 indicates it's false
        # There will be n-1 columns: 1 for each n options, minus the first option which is represented by 0s in all the other columns,
        # which ensures all the columns are independent
        use_one_hot_encoding = True

    return use_one_hot_encoding

def preprocess_result_data(result_data, problem_type):
    """
    Preprocesses the result data.

    If the problem type is classification, converts the result data to strings, creates a map from result categories
    to integers, and then converts the result data to integers.

    Args:
        result_data (pandas.Series): The result data to preprocess.
        problem_type (str): The type of problem. Either 'classification' or 'regression'.

    Returns:
        result_data (pandas.Series): The preprocessed result data.
        result_categories_map (dict): A map from result categories to integers, if applicable.
    """
    result_categories_map = None

    if problem_type == 'classification':
        result_data = result_data.astype(dtype = 'str', copy=True)
        result_categories_map = {option: value for value, option in enumerate(OrderedSet(result_data))}
        result_data = result_data.map(result_categories_map)

    return result_data, result_categories_map
