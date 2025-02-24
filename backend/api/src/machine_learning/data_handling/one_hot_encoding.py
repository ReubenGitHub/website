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