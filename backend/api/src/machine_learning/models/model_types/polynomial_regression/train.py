import numpy

def train_polynomial_regression(train_feature_data_scaled, train_result_data, poly_degree):
    """
    Trains a polynomial regression model of the given degree.

    Parameters
    ----------
    train_feature_data_scaled : array-like
        The scaled feature data to use for training.
    train_result_data : array-like
        The result data to use for training.
    poly_degree : int
        The degree of the polynomial to fit.

    Returns
    -------
    poly_reg_model : numpy.poly1d
        The trained polynomial regression model.
    """
    poly_reg_model = numpy.poly1d(numpy.polyfit(train_feature_data_scaled.ravel(), train_result_data, poly_degree))

    # Assign a predict method to the model for consistency with other models
    poly_reg_model.predict = lambda prediction_point: poly_reg_model(prediction_point)[0]

    return poly_reg_model
