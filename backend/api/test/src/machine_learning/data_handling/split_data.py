import unittest
import pandas
from api.src.machine_learning.data_handling.split_data import split_data_into_train_and_test

FEATURE_DATA = pandas.DataFrame({
    'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'B': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'C': [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
})
RESULT_DATA = pandas.DataFrame({
    'RESULT': [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
})
FEATURE_DATA_COPY = FEATURE_DATA.copy()
RESULT_DATA_COPY = RESULT_DATA.copy()

class TestSplitDataIntoTrainAndTest(unittest.TestCase):
    def test_split_data_to_target_proportion(self):
        """
        Test that splitting the data with appropriate dataset and proportion size results in the expected train and test sizes.
        """
        dataset_size = 10
        feature_data = FEATURE_DATA.head(dataset_size)
        result_data = RESULT_DATA.head(dataset_size)
        test_proportion = 0.2
        train_proportion = 1 - test_proportion

        # Split the data
        train_feature_data, test_feature_data, train_result_data, test_result_data = split_data_into_train_and_test(feature_data, result_data, test_proportion)

        # Check the train and test data sizes are as expected
        expected_train_size = dataset_size * train_proportion
        expected_test_size = dataset_size * test_proportion
        self._assert_expected_data_size(expected_train_size, expected_test_size, train_feature_data, test_feature_data, train_result_data, test_result_data)

        # Check that the original data is unchanged
        self._assert_unchanged_data(feature_data, result_data, dataset_size)

    def test_split_data_to_min_test_size(self):
        """
        Test that splitting the data with a high test proportion results in the test size being limited to the minimum.
        """
        dataset_size = 10
        feature_data = FEATURE_DATA.head(dataset_size)
        result_data = RESULT_DATA.head(dataset_size)
        test_proportion = 0.1

        # Split the data
        train_feature_data, test_feature_data, train_result_data, test_result_data = split_data_into_train_and_test(feature_data, result_data, test_proportion)

        # Check the train and test data sizes are as expected
        expected_train_size = 8
        expected_test_size = 2
        self._assert_expected_data_size(expected_train_size, expected_test_size, train_feature_data, test_feature_data, train_result_data, test_result_data)

        # Check that the original data is unchanged
        self._assert_unchanged_data(feature_data, result_data, dataset_size)

    def test_split_data_to_max_test_size(self):
        """
        Test that splitting the data with a high test proportion results in the test size being limited to a maximum.
        """
        dataset_size = 10
        feature_data = FEATURE_DATA.head(dataset_size)
        result_data = RESULT_DATA.head(dataset_size)
        test_proportion = 0.9

        # Split the data
        train_feature_data, test_feature_data, train_result_data, test_result_data = split_data_into_train_and_test(feature_data, result_data, test_proportion)

        # Check the train and test data sizes are as expected
        expected_train_size = 2
        expected_test_size = 8
        self._assert_expected_data_size(expected_train_size, expected_test_size, train_feature_data, test_feature_data, train_result_data, test_result_data)

        # Check that the original data is unchanged
        self._assert_unchanged_data(feature_data, result_data, dataset_size)

    def test_split_data_with_insufficient_data(self):
        """
        Test that an error is raised when attempting to split the data with a dataset
        which is too small for a minimum train and test set
        """
        dataset_size = 3
        feature_data = FEATURE_DATA.head(dataset_size)
        result_data = RESULT_DATA.head(dataset_size)
        test_proportion = 0.5

        # Expect an error while attempting to split the data, due to insufficient quantity of data
        with self.assertRaises(ValueError):
            train_feature_data, test_feature_data, train_result_data, test_result_data = split_data_into_train_and_test(
                feature_data,
                result_data,
                test_proportion
            )

        # Check that the original data is unchanged
        self._assert_unchanged_data(feature_data, result_data, dataset_size)

    def test_split_data_with_invalid_test_proportion(self):
        """
        Test that an error is raised when attempting to split the data with an invalid test proportion (bigger than 1)
        """
        dataset_size = 10
        feature_data = FEATURE_DATA.head(dataset_size)
        result_data = RESULT_DATA.head(dataset_size)
        test_proportion = 1.2

        # Expect an error while attempting to split the data, due to invalid test proportion
        with self.assertRaises(ValueError):
            train_feature_data, test_feature_data, train_result_data, test_result_data = split_data_into_train_and_test(
                feature_data,
                result_data,
                test_proportion
            )

        # Check that the original data is unchanged
        self._assert_unchanged_data(feature_data, result_data, dataset_size)

    def _assert_expected_data_size(self, expected_train_size, expected_test_size, train_feature_data, test_feature_data, train_result_data, test_result_data):
        # Check the train and test data sizes are as expected
        self.assertEqual(len(train_feature_data), expected_train_size)
        self.assertEqual(len(test_feature_data), expected_test_size)
        self.assertEqual(len(train_result_data), expected_train_size)
        self.assertEqual(len(test_result_data), expected_test_size)

    def _assert_unchanged_data(self, feature_data, result_data, dataset_size):
        # Check that the original data is unchanged
        pandas.testing.assert_frame_equal(feature_data, FEATURE_DATA_COPY.head(dataset_size))
        pandas.testing.assert_frame_equal(result_data, RESULT_DATA_COPY.head(dataset_size))

if __name__ == '__main__':
    unittest.main()