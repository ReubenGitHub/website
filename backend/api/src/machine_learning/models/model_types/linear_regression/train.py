from sklearn.linear_model import LinearRegression

def train_linear_regression(train_feature_data_scaled, train_result_data):
    """
    Trains a Linear Regression model.

    Parameters
    ----------
    train_feature_data_scaled : pandas.DataFrame
        The scaled feature data to use for training.
    train_result_data : pandas.DataFrame
        The result data to use for training.

    Returns
    -------
    regression_model : sklearn.linear_model.LinearRegression
        The trained Linear Regression model.
    """
    regression_model = LinearRegression()
    regression_model.fit(train_feature_data_scaled, train_result_data)

    return regression_model
