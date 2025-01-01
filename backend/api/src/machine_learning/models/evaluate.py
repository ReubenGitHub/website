import numpy
from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score

def calculate_model_metrics(
    model,
    problem_type,
    train_feature_data_scaled,
    test_feature_data_scaled,
    train_result_data,
    test_result_data
):
    """
    Calculate model metrics for a given machine learning model.

    This function evaluates the performance of a trained model on both training
    and testing datasets. Depending on the problem type (classification or
    regression), it calculates different sets of metrics.

    For classification problems, it calculates accuracy, macro precision,
    micro precision, macro recall, and micro recall. For regression problems,
    it calculates the R-squared score as a measure of accuracy.

    Parameters
    ----------
    model : object
        The trained machine learning model to evaluate.
    problem_type : str
        The type of problem, either 'classification' or 'regression'.
    train_feature_data_scaled : array-like
        The scaled feature data used for training the model.
    test_feature_data_scaled : array-like
        The scaled feature data used for testing the model.
    train_result_data : array-like
        The actual result data corresponding to the training features.
    test_result_data : array-like
        The actual result data corresponding to the testing features.

    Returns
    -------
    dict
        A dictionary containing the calculated metrics, with keys indicating
        the type of metric (e.g., 'train_accuracy', 'test_macro_precision').
    """
    train_predictions = model.predict(numpy.array(train_feature_data_scaled))
    test_predictions = model.predict(numpy.array(test_feature_data_scaled))

    model_metrics = {}

    if problem_type == 'classification':
        # Calculate accuracy
        model_metrics['train_accuracy'] = accuracy_score(train_result_data, train_predictions)
        model_metrics['test_accuracy'] = accuracy_score(test_result_data, test_predictions)

        # Calculate macro precision - average of individual class precisions = numpy.average( TP / (TP+FP) )
        # Good for spotting poor precision on minority classes
        model_metrics['train_macro_precision'] = precision_score(train_result_data, train_predictions, average="macro", zero_division=0)
        model_metrics['test_macro_precision'] = precision_score(test_result_data, test_predictions, average="macro", zero_division=0)

        # Calculate micro precision - weighted average of individual class precisions = TP.sum() / (TP.sum()+FP.sum())
        # Good if you only care about precision in majority of cases
        model_metrics['train_micro_precision'] = precision_score(train_result_data, train_predictions, average="micro", zero_division=0)
        model_metrics['test_micro_precision'] = precision_score(test_result_data, test_predictions, average="micro", zero_division=0)

        # Calculate macro/micro recall
        # Micro precision/micro recall/micro F1-score are all equal to accuracy when items can only have one label each (i.e. non-Multi-Label problem)
        model_metrics['train_macro_recall'] = recall_score(train_result_data, train_predictions, average="macro", zero_division=0)
        model_metrics['test_macro_recall'] = recall_score(test_result_data, test_predictions, average="macro", zero_division=0)
        model_metrics['train_micro_recall'] = recall_score(train_result_data, train_predictions, average="micro", zero_division=0)
        model_metrics['test_micro_recall'] = recall_score(test_result_data, test_predictions, average="micro", zero_division=0)
    else: # problem_type == "regression": (always the case for LinReg and PolyFit)
        # Calculate accuracy/coefficient of determination - r-squared
        # Ideally R^2 will be similar for the training and testing data sets, meaning the model isn't overfit to the training data. 
        # When R^2<0, a horizontal line (mean) explains the data better than your model.
        model_metrics['train_accuracy'] = r2_score(train_result_data, train_predictions)
        model_metrics['test_accuracy'] = r2_score(test_result_data, test_predictions)

    return model_metrics
