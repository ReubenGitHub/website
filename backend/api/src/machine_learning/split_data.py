from sklearn.model_selection import train_test_split

# Need at least 2 test/train samples (for measuring regression accuracy)
MIN_TEST_TRAIN_SIZE = 2

def split_data_into_train_and_test(feature_data, result_data, test_proportion):
    """
    Splits feature and result data into training and testing sets based on the given test proportion.

    This function computes the size of the test data as a proportion of the total data size, ensuring
    that the test size is at least the minimum required and does not exceed the maximum allowable size.
    It then splits the data into training and testing sets using the computed test size.

    Parameters
    ----------
    feature_data : array-like
        The feature data to be split.
    result_data : array-like
        The result data to be split.
    test_proportion : float
        The proportion of the data to be used as the test set (between 0 and 1).

    Returns
    -------
    train_feature_data : array-like
        The feature data for the training set.
    test_feature_data : array-like
        The feature data for the testing set.
    train_result_data : array-like
        The result data for the training set.
    test_result_data : array-like
        The result data for the testing set.
    """
    sample_size = len(feature_data)

    min_test_size = MIN_TEST_TRAIN_SIZE
    max_test_size = sample_size - MIN_TEST_TRAIN_SIZE
    target_test_size = round(test_proportion * sample_size)
    test_size = min(max(min_test_size, target_test_size), max_test_size)

    train_feature_data, test_feature_data, train_result_data, test_result_data = train_test_split(feature_data, result_data, test_size = test_size)

    return train_feature_data, test_feature_data, train_result_data, test_result_data