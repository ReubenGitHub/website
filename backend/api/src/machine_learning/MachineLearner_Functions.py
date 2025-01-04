from .session_context_manager import session_context_manager
from .data_handling.preprocess_data import preprocess_data
from .data_handling.split_data import split_data_into_train_and_test
from .data_handling.scale_data import scale_data
from .models.train_model import train_model
from .models.evaluate import calculate_model_metrics
from .models.graphing.generate_image import generate_model_graph
from .models.graphing.graph_types.one_d_function_plot import generate_1d_function_plot
from .models.graphing.graph_types.two_d_function_plot import generate_2d_function_plot
from .models.graphing.graph_types.two_d_region_plot import generate_2d_region_plot
from .models.graphing.graph_types.tree_graph import generate_tree_graph
import numpy
import matplotlib
matplotlib.use('Agg')
# from scipy import stats
import pandas
from contextlib import contextmanager
# import threading
# import _thread
# import multiprocessing

# class TimeoutException(Exception):
#     def __init__(self, msg=''):
#         self.msg = msg

# class MyException(Exception):
#     pass

# @contextmanager
# def time_limit(cwd, seconds, msg=''):
# # def time_limit(supervision, problem_type, model_type, poly_degree, continuous_features, categorical_features, result, test_proportion, datasetName, cwd, seconds, msg=''):
#     # timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
#     timer = threading.Timer( seconds, lambda: (print("TIMED OUT SO CANCELLED"), MyException) )
#     timer.start()
#     print("TIMER STARTED AND IS")
#     try:
#         print("TRYING")
#         # yield
#         yield
#     except MyException:
#         print("TIMED OUT SO CANCELLED")
#         raise TimeoutException("Timed out for operation {}".format(msg)) 
#     finally:
#         timer.cancel()
#         print("STOPPED?")

# def machineLearnerTimed(supervision, problem_type, model_type, poly_degree, continuous_features, categorical_features, result, test_proportion, datasetName):
#     cwd = os.getcwd()
#     try:
#         print("KICKING OFF ML WITH TIME LIMIT")
#         with time_limit(cwd, 5, 'sleep'):
#         # return time_limit(supervision, problem_type, model_type, poly_degree, continuous_features, categorical_features, result, test_proportion, datasetName, cwd, 5, 'sleep')
#             return machineLearner(supervision, problem_type, model_type, poly_degree, continuous_features, categorical_features, result, test_proportion, datasetName)
#     except TimeoutException:
#         print("TIME RAN OUT EXCEPTION, RETURNING NOTHING")
#         return
#     print("HERE 4")


def machineLearner(supervision, problem_type, model_type, poly_degree, continuous_features, categorical_features, result, test_proportion, session_id): #, accTrain, accTest, inpVal):
    # Disable pandas warning "A value is trying to be set on a copy of a slice from a DataFrame"
    pandas.options.mode.chained_assignment = None  # default='warn'

    #Method of learning: DT, LinReg, PolyFit (1d), ...  . poly_degree degress of polynomial if PolyFit selected
    #Features to analyse, independent X, and proportion of test vs train. Cts followed by categorical
    features = continuous_features + categorical_features
    fields_of_interest = features + [result]
    global ncts
    global ncate
    global ntotal
    ncts = len(continuous_features)
    ncate = len(categorical_features)
    ntotal = ncts + ncate

    # Get dataset from session data
    dataset = session_context_manager.get_session_data(session_id)['dataset']
    dataset = dataset.dropna(axis=0, how='any', thresh=None, subset=fields_of_interest, inplace=False)

    # Preprocess data to split out feature and result data, and handle category encoding (for categorical features or results)
    feature_data, categorical_features_category_maps, result_data, result_categories_map = preprocess_data(
        dataset, categorical_features, continuous_features, result, model_type, problem_type
    )

    # Split data into test and train sets
    train_feature_data, test_feature_data, train_result_data, test_result_data = split_data_into_train_and_test(
        feature_data, result_data, test_proportion
    )

    # Apply scaling to continuous features if required
    train_feature_data_scaled, test_feature_data_scaled, scale = scale_data(
        train_feature_data, test_feature_data, continuous_features, model_type
    )

    # If scaling was applied, convert data to numpy arrays (legacy support)
    # Todo: work out why this was done and if there's a nicer way to handle this
    if (scale is not None):
        train_feature_data_scaled = train_feature_data_scaled.to_numpy()
        test_feature_data_scaled = test_feature_data_scaled.to_numpy()

    # Todo: tidy this up
    if (model_type == 'LinReg' and len(continuous_features) == 0):
        train_feature_data_scaled = train_feature_data_scaled.to_numpy()
        test_feature_data_scaled = test_feature_data_scaled.to_numpy()

    # Train model
    training_parameters = {
        'problem_type': problem_type,
        'num_continuous_features': len(continuous_features),
        'num_categorical_features': len(categorical_features),
        'poly_degree': poly_degree
    }
    model = train_model(train_feature_data_scaled, train_result_data, model_type, training_parameters)
    session_context_manager.add_model(session_id, model)

    # Calculate model metrics (accuracy etc...)
    model_metrics = calculate_model_metrics(
        model,
        problem_type,
        train_feature_data_scaled,
        test_feature_data_scaled,
        train_result_data,
        test_result_data
    )

    # Generate model graph
    graph_image_base_64 = generate_model_graph(
        model,
        model_type,
        problem_type,
        continuous_features,
        categorical_features,
        result,
        train_feature_data,
        test_feature_data,
        train_result_data,
        test_result_data,
        scale,
        result_categories_map
    )

    print("REPRESENTATION CREATED AND SAVED, NOW SAVING MODEL SETTINGS/PARAMS")
    modelSettings = {
        'probType': problem_type,
        'model_type': model_type,
        'features': features,
        'continuous_features': continuous_features,
        'categorical_features': categorical_features,
        'categorical_features_category_maps': categorical_features_category_maps,
        'result_categories_map': result_categories_map,
        'dataset': dataset,
        'scaler': scale
    }

    session_context_manager.add_model_settings(session_id, modelSettings)

    print("SETTINGS SAVED, NOW SETTING UP INPUT VALIDATION AND EXPORTING")

    # Create input validation object
    inputValidation = []
    for i in range(0, len(categorical_features)):
        inputValidation.append(list(categorical_features_category_maps[categorical_features[i]].keys()))

    # Define outputs
    outputs = {
        'model_metrics': model_metrics,
        'graph_image_base_64': graph_image_base_64
    }

    return {'mlOuts': outputs, 'inputValidation': inputValidation}

# def machineLearnerTimed(supervision, problem_type, model_type, poly_degree, continuous_features, categorical_features, result, test_proportion, datasetName):
#     cwd = os.getcwd()

#     accTrain = multiprocessing.Value('d')
#     accTest = multiprocessing.Value('d')
#     inpVal = multiprocessing.Array('u', [])
    
#     print("STARTING TIMED FIT THREAD")
#     # p = threading.Thread(target = machineLearner, args = [supervision, problem_type, model_type, poly_degree, continuous_features, categorical_features, result, test_proportion, datasetName])
#     p = multiprocessing.Process(target = machineLearner, args = [supervision, problem_type, model_type, poly_degree, continuous_features, categorical_features, result, test_proportion, datasetName, accTrain, accTest, inpVal])
#     p.start()
#     p.join(5)

#     if p.is_alive():
#         print("Terminating fit thread")
#         p.terminate()
#         p.join()
#         return
#     else:
#         print(accTrain)
#         print(accTest)
#         print(inpVal)
#         return {"accuracyTrain": accTrain, "accuracyTest": accTest, "inputValidation": inpVal}

#Evaluate model at some particular value
def modelPrediction(predictAt, session_id):
    session_data = session_context_manager.get_session_data(session_id)
    model = session_data['model']
    modelSettings = session_data['model_settings']

    problem_type = modelSettings['probType']
    model_type = modelSettings['model_type']
    features = modelSettings['features']
    continuous_features = modelSettings['continuous_features']
    categorical_features = modelSettings['categorical_features']
    categorical_features_category_maps = modelSettings['categorical_features_category_maps']
    result_categories_map = modelSettings['result_categories_map']
    dataset = modelSettings['dataset']
    scale = modelSettings['scaler']

    if model_type == 'DT':
        predictAtNumerical = predictAt.copy()
        for i in range(0,len(features)):
            if features[i] in categorical_features:
                category = features[i]
                predictAtNumerical[i] = categorical_features_category_maps[category][predictAtNumerical[i]]
        predicted = model.predict([predictAtNumerical])
        if problem_type == 'classification':
            predicted = [list(result_categories_map.keys())[predicted[0]]]
    elif model_type == 'KNN':
        predictAtAdj = []
        if len(continuous_features) > 0:
            predictAtAdj[0:len(continuous_features)] = scale.transform([predictAt[0:len(continuous_features)]]).ravel()
        if len(categorical_features) > 0:
            for i in range(0,len(categorical_features)):
                category = features[len(continuous_features) + i]
                predictAtAdj.append( categorical_features_category_maps[category][predictAt[len(continuous_features) + i]] )
        predicted = model.predict(numpy.array(predictAtAdj).reshape(1, -1))
        if problem_type == 'classification':
            predicted = [list(result_categories_map.keys())[predicted[0]]]
    elif model_type == 'LinReg':
        scaled = []
        predictCates = []
        if len(continuous_features) > 0:
            scaled = scale.transform([predictAt[0:len(continuous_features)]])
            scaled = scaled.ravel()
        if len(categorical_features) > 0:
            for i in range(0,len(categorical_features)):
                dummyCate = [0] * ( len(set(dataset[categorical_features[i]])) - 1 )
                onePos = sorted(set(dataset[categorical_features[i]])).index(predictAt[len(continuous_features)+i]) - 1
                if onePos > -1:
                    dummyCate[onePos] = 1
                predictCates = predictCates + dummyCate
        predictAtAdj = [*scaled, *predictCates]
        predicted = model.predict([predictAtAdj])
    elif model_type == "PolyFit":
        scaled = scale.transform([predictAt])
        predicted = model.predict(scaled)

    #POSSIBLY ROUND Predicted 
    #round_to_n = lambda x, n: round(x, -int(floor(log10(x))) + (n - 1))
    
    return {"predictAt": predictAt, "prediction": predicted[0]}


# Ensure this is called when the website is closed or page refreshed
def clear_session_data(session_id):
    session_context_manager.remove_session_data(session_id)
    return
