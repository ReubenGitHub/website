import unittest
import pandas
from api.src.machine_learning.train_models.train_model import train_model
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from numpy import poly1d

TRAIN_FEATURE_DATA = pandas.DataFrame({
    'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'B': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'C': [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
})
TRAIN_RESULT_DATA = pandas.DataFrame({
    'RESULT': [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
})
TRAIN_FEATURE_DATA_COPY = TRAIN_FEATURE_DATA.copy()
TRAIN_RESULT_DATA_COPY = TRAIN_RESULT_DATA.copy()

# Todo: Update use of this to be centralized somewhere
TRAIN_FEATURE_DATA_NUMPY = TRAIN_FEATURE_DATA.to_numpy()
TRAIN_RESULT_DATA_NUMPY = TRAIN_RESULT_DATA.to_numpy()

NUM_CONTINUOUS_FEATURES = 2
NUM_CATEGORICAL_FEATURES = 1

PREDICTION_POINT = TRAIN_FEATURE_DATA.iloc[0].to_frame().transpose()
EXPECTED_PREDICTION = TRAIN_RESULT_DATA.iloc[0].values[0]

class TestTrainModel(unittest.TestCase):
    def test_train_decision_tree_classification(self):
        """
        Test that a Decision Tree model for classification can be trained successfully.
        """
        model_type = 'DT'
        training_parameters = { 'problem_type': 'classification' }

        # Train the model
        model = train_model(TRAIN_FEATURE_DATA, TRAIN_RESULT_DATA, model_type, training_parameters)

        # Check that the model is an instance of the expected class
        self.assertIsInstance(model, DecisionTreeClassifier)
        
        # Check the model has a working "predict" method
        self._assert_prediction_accuracy(model)

        # Check that input data are unchanged
        self._assert_unchanged_data()

    def test_train_decision_tree_regression(self):
        """
        Test that a Decision Tree model for regression can be trained successfully.
        """
        model_type = 'DT'
        training_parameters = { 'problem_type': 'regression' }

        # Train the model
        model = train_model(TRAIN_FEATURE_DATA, TRAIN_RESULT_DATA, model_type, training_parameters)

        # Check that the model is an instance of the expected class
        self.assertIsInstance(model, DecisionTreeRegressor)
        
        # Check the model has a working "predict" method
        self._assert_prediction_accuracy(model)

        # Check that input data are unchanged
        self._assert_unchanged_data()

    def test_train_knn_classification(self):
        """
        Test that a K-Nearest Neighbours model for classification can be trained successfully.
        """
        model_type = 'KNN'
        training_parameters = {
            'problem_type': 'classification',
            'num_continuous_features': NUM_CONTINUOUS_FEATURES,
            'num_categorical_features': NUM_CATEGORICAL_FEATURES
        }

        # Train the model
        model = train_model(TRAIN_FEATURE_DATA, TRAIN_RESULT_DATA, model_type, training_parameters)

        # Check that the model is an instance of the expected class
        self.assertIsInstance(model, KNeighborsClassifier)

        # Check the model has a working "predict" method
        self._assert_prediction_accuracy(model)

        # Check that input data are unchanged
        self._assert_unchanged_data()

    def test_train_knn_regression(self):
        """
        Test that a K-Nearest Neighbours model for regression can be trained successfully.
        """
        model_type = 'KNN'
        training_parameters = {
            'problem_type': 'regression',
            'num_continuous_features': NUM_CONTINUOUS_FEATURES,
            'num_categorical_features': NUM_CATEGORICAL_FEATURES
        }

        # Train the model
        model = train_model(TRAIN_FEATURE_DATA, TRAIN_RESULT_DATA, model_type, training_parameters)

        # Check that the model is an instance of the expected class
        self.assertIsInstance(model, KNeighborsRegressor)

        # Check the model has a working "predict" method
        self._assert_prediction_accuracy(model)

        # Check that input data are unchanged
        self._assert_unchanged_data()

    def test_train_linear(self):
        """
        Test that a linear regression model can be trained successfully.
        """
        model_type = 'LinReg'

        # Todo: Update use of this to be centralized somewhere
        train_feature_data_numpy = TRAIN_FEATURE_DATA.to_numpy()

        # Train the model
        model = train_model(train_feature_data_numpy, TRAIN_RESULT_DATA, model_type)

        # Check that the model is an instance of the expected class
        self.assertIsInstance(model, LinearRegression)

        # Check the model has a working "predict" method
        self._assert_prediction_accuracy(model)

        # Check that input data are unchanged
        self._assert_unchanged_data()

    def test_train_polynomial(self):
        """
        Test that a polynomial regression model can be trained successfully.
        """
        model_type = 'PolyFit'
        training_parameters = { 'poly_degree': 2 }

        # Polynomial regression model only works with 1D data
        # Todo: Update array-ifying dataframes to be centralized somewhere
        train_feature_data_numpy = TRAIN_FEATURE_DATA.iloc[:, 0].to_numpy().reshape(-1, 1)
        train_result_data_array = TRAIN_RESULT_DATA.to_numpy().ravel()

        # Train the model
        model = train_model(train_feature_data_numpy, train_result_data_array, model_type, training_parameters)

        # Check that the model is an instance of the expected class
        self.assertIsInstance(model, poly1d)

        # Check the model has a working "predict" method
        # Polynomial regression model only works with 1D data
        prediction_point = [PREDICTION_POINT.iloc[0, 0]]
        self._assert_prediction_accuracy(model, prediction_point)

        # Check that input data are unchanged
        self._assert_unchanged_data()

    def _assert_unchanged_data(self):
        """
        Asserts that the original training feature and result data have not been modified.
        """
        pandas.testing.assert_frame_equal(TRAIN_FEATURE_DATA, TRAIN_FEATURE_DATA_COPY)
        pandas.testing.assert_frame_equal(TRAIN_RESULT_DATA, TRAIN_RESULT_DATA_COPY)

    def _assert_prediction_accuracy(self, model, prediction_point=PREDICTION_POINT):
        """
        Asserts that the model's prediction accuracy is within an acceptable delta of the expected value.
        """
        prediction = model.predict(prediction_point)
        self.assertAlmostEqual(prediction, EXPECTED_PREDICTION, delta=1)

    def test_invalid_model_type(self):
        """
        Test that a ValueError is raised when attempting to train a model with an invalid type.
        """
        model_type = 'InvalidModelType'

        # Check that a ValueError is raised
        with self.assertRaises(ValueError):
            model = train_model(TRAIN_FEATURE_DATA, TRAIN_RESULT_DATA, model_type)

if __name__ == '__main__':
    unittest.main()