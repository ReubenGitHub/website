import unittest
import pandas
from sklearn.preprocessing import StandardScaler
from api.src.machine_learning.scale_data import scale_data

TRAIN_FEATURE_DATA = pandas.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [2, 4, 2],
})
TEST_FEATURE_DATA = pandas.DataFrame({
    'A': [7, 8, 9],
    'B': [10, 11, 12],
    'C': [1, 2, 3],
})
TRAIN_FEATURE_DATA_COPY = TRAIN_FEATURE_DATA.copy()
TEST_FEATURE_DATA_COPY = TEST_FEATURE_DATA.copy()
CONTINUOUS_FEATURES = ['A', 'B']
MODEL_TYPES_WITH_SCALING = ['LinReg', 'PolyFit', 'KNN']

class TestScaleData(unittest.TestCase):
    def test_scale_data_do_scaling(self):
        """
        Test that scale_data scales data correctly when it should.
        """
        for model_type in MODEL_TYPES_WITH_SCALING:
            # Scale the data
            train_feature_data_scaled, test_feature_data_scaled, scale = scale_data(TRAIN_FEATURE_DATA, TEST_FEATURE_DATA, CONTINUOUS_FEATURES, model_type)
            # Check that all the means (of all continuous features) are 0
            means = [train_feature_data_scaled[feature].mean() for feature in CONTINUOUS_FEATURES]
            self.assertAlmostEqual(means[0], 0, places=4)

            # Check that all the standard deviations (of all continuous features) are equal
            stds = [train_feature_data_scaled[feature].std() for feature in CONTINUOUS_FEATURES]
            self.assertTrue(all(abs(std - stds[0]) < 0.001 for std in stds))

            # Check that the scale is a scale, and that the scale means match the original data means
            # StandardScaler uses the "biased" std and Pandas uses the "unbiased" std, so don't check scale std against data std
            self.assertIsInstance(scale, StandardScaler)
            for i, feature in enumerate(CONTINUOUS_FEATURES):
                self.assertAlmostEqual(scale.mean_[i], TRAIN_FEATURE_DATA[feature].mean(), places=4)
            
            # Check that non-continuous data and input data are unchanged
            self._assert_unchanged_data(train_feature_data_scaled)

    def test_scale_data_no_scaling(self):
        """
        Test that scale_data does not scale data when it shouldn't.
        """
        model_type = 'DT'
        train_feature_data_scaled, test_feature_data_scaled, scale = scale_data(TRAIN_FEATURE_DATA, TEST_FEATURE_DATA, CONTINUOUS_FEATURES, model_type)

        # Check that scaling hasn't been applied and data is unchanged
        pandas.testing.assert_frame_equal(train_feature_data_scaled, TRAIN_FEATURE_DATA)
        pandas.testing.assert_frame_equal(test_feature_data_scaled, TEST_FEATURE_DATA)
        self.assertIsNone(scale)

        # Check that non-continuous data and input data are unchanged
        self._assert_unchanged_data(train_feature_data_scaled)

    def _assert_unchanged_data(self, train_feature_data_scaled):
        """
        Asserts that non-continuous features and the original input data remain unchanged.
        """
        # Check that non-continuous data is unchanged
        pandas.testing.assert_frame_equal(
            train_feature_data_scaled.drop(columns=CONTINUOUS_FEATURES),
            TRAIN_FEATURE_DATA.drop(columns=CONTINUOUS_FEATURES)
        )

        # Check that input data is unchanged
        pandas.testing.assert_frame_equal(TRAIN_FEATURE_DATA, TRAIN_FEATURE_DATA_COPY)
        pandas.testing.assert_frame_equal(TEST_FEATURE_DATA, TEST_FEATURE_DATA_COPY)

if __name__ == '__main__':
    unittest.main()
