from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

PROBLEM_TYPE_TO_CLASS = {
    'classification': DecisionTreeClassifier,
    'regression': DecisionTreeRegressor
}

def train_decision_tree(train_feature_data_scaled, train_result_data, problem_type):
    """
    Trains a decision tree on the given data.

    Parameters
    ----------
    train_feature_data_scaled : pandas.DataFrame
        The scaled feature data to use for training.
    train_result_data : pandas.DataFrame
        The result data to use for training.
    problem_type : str
        The type of problem to train for, either 'classification' or 'regression'.

    Returns
    -------
    decision_tree : sklearn.tree.DecisionTreeClassifier or DecisionTreeRegressor
        The trained decision tree model.
    """
    # Gives warning that we're training with feature names, but this is fine for now
    decision_tree_class = PROBLEM_TYPE_TO_CLASS[problem_type]()
    decision_tree = decision_tree_class.fit(train_feature_data_scaled.values, train_result_data.values)

    return decision_tree
