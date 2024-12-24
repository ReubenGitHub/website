from sklearn.preprocessing import StandardScaler
import pandas

def scale_data(train_feature_data, test_feature_data, continuous_features, model_type):
    """
    Scales the feature data of the training and test datasets.

    This function takes feature data from the training and test datasets, and scales the
    data according to the type of model to be used. It will only scale the continuous
    features, and will not modify the categorical features.

    NOTE: This function will change the indices of the scaled training and test feature
    data compared to the original

    Parameters
    ----------
    train_feature_data : pandas.DataFrame
        The feature data from the training dataset.
    test_feature_data : pandas.DataFrame
        The feature data from the test dataset.
    continuous_features : list
        A list of the column names of the continuous features.
    model_type : str
        The type of model to use. Can be 'DT', 'KNN', 'LinReg', 'PolyFit', etc.

    Returns
    -------
    train_feature_data_scaled : pandas.DataFrame
        The scaled feature data for the training dataset.
    test_feature_data_scaled : pandas.DataFrame
        The scaled feature data for the test dataset.
    scale : sklearn.preprocessing.StandardScaler
        The scaler object used to scale the data. If no scaling was required, this will be None.
    """
    # Avoid mutating the original data either in this function or in calling code
    train_feature_data_scaled = train_feature_data.copy()
    test_feature_data_scaled = test_feature_data.copy()

    # Ensure scale is defined
    scale = None

    # If there are continuous features and scaling is required due to the model type
    if len(continuous_features) > 0 and (model_type in ['LinReg', 'PolyFit', 'KNN']):
        scale = StandardScaler()

        # Scale data as defined by the training dataset, then apply to the test data
        train_feature_data_scaled_array = scale.fit_transform(train_feature_data[continuous_features].values)
        test_feature_data_scaled_array = scale.transform(test_feature_data[continuous_features].values)

        # Must reset indices before assigning scaled data as scaled arrays have no indices
        train_feature_data_scaled.reset_index(drop=True, inplace=True)
        test_feature_data_scaled.reset_index(drop=True, inplace=True)

        # Overwrite the continuous feature data with the scaled data
        train_feature_data_scaled[continuous_features] = pandas.DataFrame(
            train_feature_data_scaled_array,
            columns=continuous_features
        )
        test_feature_data_scaled[continuous_features] = pandas.DataFrame(
            test_feature_data_scaled_array,
            columns=continuous_features
        )

        # TODO: Move where this is done, so that the output of this code is just the scaled data in a dataframe. What does this operation do to the indices (relative to the results)?
        train_feature_data_scaled = train_feature_data_scaled.to_numpy()
        test_feature_data_scaled = test_feature_data_scaled.to_numpy()
        # TODO: Check whether this is required
        # train_result_data = train_result_data.reset_index(drop=True)
        # test_result_data = test_result_data.reset_index(drop=True)

    return train_feature_data_scaled, test_feature_data_scaled, scale
