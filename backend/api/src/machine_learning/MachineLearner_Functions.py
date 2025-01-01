from .session_context_manager import session_context_manager
from .data_handling.preprocess_data import preprocess_data
from .data_handling.split_data import split_data_into_train_and_test
from .data_handling.scale_data import scale_data
from .models.train_model import train_model
from .models.evaluate import calculate_model_metrics
from xml.dom.minicompat import StringTypes
import numpy
import matplotlib
from sklearn.externals._packaging.version import _parse_version_parts
matplotlib.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.image as pltimg3
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# from scipy import stats
from sklearn import tree
import pandas
import pydotplus
import colorsys
from contextlib import contextmanager
import io
import base64
import re
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

    # Default initialization to avoid checking for non-existence later
    repImageBase64 = None

    # Draw/plot the model tree/3d training data
    repImageCreated=True
    if model_type == "DT":
        if problem_type == "classification":
            if len(continuous_features) == 2 and len(categorical_features) == 0:
                train_result_data = train_result_data.reset_index(drop=True)
                test_result_data = test_result_data.reset_index(drop=True)
                xsTest = test_feature_data.iloc[:, 0].reset_index(drop=True)
                ysTest = test_feature_data.iloc[:, 1].reset_index(drop=True)
                xsTrain = train_feature_data.iloc[:, 0].reset_index(drop=True)
                ysTrain = train_feature_data.iloc[:, 1].reset_index(drop=True)
                xs = feature_data.iloc[:, 0]
                ys = feature_data.iloc[:, 1]
                xs = numpy.linspace(int(numpy.floor(min(xs)))-1,int(numpy.ceil(max(xs)))+1,200)
                ys = numpy.linspace(int(numpy.floor(min(ys)))-1,int(numpy.ceil(max(ys)))+1,200)
                xs, ys = numpy.meshgrid( xs, ys )
                coords = numpy.vstack([xs.ravel(), ys.ravel()]).transpose()
                zs = model.predict(coords)
                lightcmap = cm.get_cmap('rainbow', len(result_categories_map))
                darkcmap = cm.get_cmap('rainbow', len(result_categories_map))
                lightcolours=[]
                listOfzs = list(set(zs))
                listOfzs.sort()
                for ind in range(len(result_categories_map)):
                    c = colorsys.rgb_to_hls(lightcmap(ind)[0], lightcmap(ind)[1], lightcmap(ind)[2])
                    lightcolours.append( tuple(colorsys.hls_to_rgb(c[0], 1 - 0.5 * (1 - c[1]), c[2]*0.7)) )
                lightcmap = cm.colors.ListedColormap(lightcolours)
                zs = zs.reshape(xs.shape)
                plt.pcolormesh(xs, ys, zs, cmap=lightcmap, vmin=min(result_categories_map.values()), vmax=max(result_categories_map.values()))
                count = 0
                for res in result_categories_map:
                    count += 1
                    dictRes = result_categories_map[res]
                    indices = numpy.where(train_result_data == dictRes)[0]
                    plt.scatter(xsTrain[indices], ysTrain[indices], color=darkcmap(dictRes), marker='o', label=(count>18)*"_"+res[0:8])
                    indices = numpy.where(test_result_data == dictRes)[0]
                    plt.scatter(xsTest[indices], ysTest[indices], color=darkcmap(dictRes), marker='x')
                plt.xlabel(train_feature_data.columns[0])
                plt.ylabel(train_feature_data.columns[1])
                plt.legend(bbox_to_anchor=(1,1), loc="upper left")
                repImageBase64 = encodeRepImage('plt', plt, format='png')
            else:
                data = tree.export_graphviz(model, out_file=None, feature_names=features, class_names=list(result_categories_map), max_depth=4, filled=True)
                graph = pydotplus.graph_from_dot_data(data)
                removeValuesFromNodes(graph)
                repImageBase64 = encodeRepImage('graph', graph)
        elif problem_type == "regression":
            if len(continuous_features) == 1 and len(categorical_features) == 0:
                plt.scatter(train_feature_data, train_result_data, marker='o', color='#03bffe', label='Training Data')
                xs = feature_data.iloc[:, 0]
                xs = numpy.linspace(int(numpy.floor(min(xs)))-1,int(numpy.ceil(max(xs)))+1,200).reshape(-1,1)
                zs = model.predict(xs)
                plt.xlabel(train_feature_data.columns[0])
                plt.ylabel(result)
                plt.plot( xs, zs, color="#fe4203", label='Model')
                plt.scatter(test_feature_data, test_result_data, marker='x', color='#ff845b', label='Testing Data')
                plt.legend(bbox_to_anchor=(0.5,1.1), loc="upper center", ncol=3)
                repImageBase64 = encodeRepImage('plt', plt, format='png')
            elif len(continuous_features) == 2 and len(categorical_features) == 0:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                xs = train_feature_data.iloc[:, 0].reset_index(drop=True)
                ys = train_feature_data.iloc[:, 1].reset_index(drop=True)
                zs = train_result_data.reset_index(drop=True)
                ax.scatter(xs, ys, zs, marker='o', color='#03bffe', label='Training Data')
                xs = feature_data.iloc[:, 0]
                ys = feature_data.iloc[:, 1]
                xs = numpy.linspace(int(numpy.floor(min(xs)))-1,int(numpy.ceil(max(xs)))+1,200)
                ys = numpy.linspace(int(numpy.floor(min(ys)))-1,int(numpy.ceil(max(ys)))+1,200)
                xs, ys = numpy.meshgrid( xs, ys )
                coords = numpy.vstack([xs.ravel(), ys.ravel()]).transpose()
                zs = model.predict(coords)
                zs = zs.reshape(xs.shape)
                surf = ax.plot_surface(xs, ys, zs, alpha = 0.5, cmap='copper', color="#03bffe", label='Model')
                #plot 3d testing data
                xs = test_feature_data.iloc[:, 0].reset_index(drop=True)
                ys = test_feature_data.iloc[:, 1].reset_index(drop=True)
                zs = test_result_data.reset_index(drop=True)
                ax.scatter(xs, ys, zs, marker='x', color='#ff845b', label='Testing Data')
                ax.set_xlabel(train_feature_data.columns[0])
                ax.set_ylabel(train_feature_data.columns[1])
                ax.set_zlabel(result)
                surf._edgecolors2d = surf._edgecolor3d
                surf._facecolors2d = surf._facecolor3d
                ax.legend(bbox_to_anchor=(0.5,1.1), loc="upper center", ncol=3)
                repImageBase64 = encodeRepImage('plt', plt, format='png')
            else:
                data = tree.export_graphviz(model, out_file=None, feature_names=features, max_depth=4, filled=True)
                graph = pydotplus.graph_from_dot_data(data)
                removeValuesFromNodes(graph)
                repImageBase64 = encodeRepImage('graph', graph)
        else:
            repImageCreated=False
    elif model_type == "KNN":
        if problem_type == "classification" and len(continuous_features) == 2 and len(categorical_features) == 0:
            train_result_data = train_result_data.reset_index(drop=True)
            test_result_data = test_result_data.reset_index(drop=True)
            xsTest = test_feature_data.iloc[:, 0].reset_index(drop=True)
            ysTest = test_feature_data.iloc[:, 1].reset_index(drop=True)
            xsTrain = train_feature_data.iloc[:, 0].reset_index(drop=True)
            ysTrain = train_feature_data.iloc[:, 1].reset_index(drop=True)
            xs = feature_data.iloc[:, 0]
            ys = feature_data.iloc[:, 1]
            xs = numpy.linspace(int(numpy.floor(min(xs)))-1,int(numpy.ceil(max(xs)))+1,200)
            ys = numpy.linspace(int(numpy.floor(min(ys)))-1,int(numpy.ceil(max(ys)))+1,200)
            xs, ys = numpy.meshgrid( xs, ys )
            coords = numpy.vstack([xs.ravel(), ys.ravel()])
            coordsScaled = scale.transform( coords.transpose() )
            zs = model.predict( coordsScaled )
            lightcmap = cm.get_cmap('rainbow', len(result_categories_map))
            darkcmap = cm.get_cmap('rainbow', len(result_categories_map))
            lightcolours=[]
            listOfzs = list(set(zs))
            listOfzs.sort()
            for ind in range(len(result_categories_map)):
                c = colorsys.rgb_to_hls(lightcmap(ind)[0], lightcmap(ind)[1], lightcmap(ind)[2])
                lightcolours.append( tuple(colorsys.hls_to_rgb(c[0], 1 - 0.5 * (1 - c[1]), c[2]*0.7)) )
            lightcmap = cm.colors.ListedColormap(lightcolours)
            zs = zs.reshape(xs.shape)
            plt.pcolormesh(xs, ys, zs, cmap=lightcmap, vmin=min(result_categories_map.values()), vmax=max(result_categories_map.values()))
            count = 0
            for res in result_categories_map:
                count += 1
                dictRes = result_categories_map[res]
                indices = numpy.where(train_result_data == dictRes)[0]
                plt.scatter(xsTrain[indices], ysTrain[indices], color=darkcmap(dictRes), marker='o', label=(count>18)*"_"+res[0:8])
                indices = numpy.where(test_result_data == dictRes)[0]
                plt.scatter(xsTest[indices], ysTest[indices], color=darkcmap(dictRes), marker='x')
            plt.xlabel(train_feature_data.columns[0])
            plt.ylabel(train_feature_data.columns[1])
            plt.legend(bbox_to_anchor=(1,1), loc="upper left")
            repImageBase64 = encodeRepImage('plt', plt, format='png')
        elif problem_type == "regression" and len(continuous_features) == 1 and len(categorical_features) == 0:
            plt.scatter(train_feature_data, train_result_data, marker='o', color='#03bffe', label='Training Data')
            xs = feature_data.iloc[:, 0]
            xs = numpy.linspace(int(numpy.floor(min(xs)))-1,int(numpy.ceil(max(xs)))+1,200)
            xsScaled = scale.transform(xs.reshape(-1,1))
            zs = model.predict( xsScaled )
            plt.xlabel(train_feature_data.columns[0])
            plt.ylabel(result)
            plt.plot( xs, zs, color="#fe4203", label='Model')
            plt.scatter(test_feature_data, test_result_data, marker='x', color='#ff845b', label='Testing Data')
            plt.legend(bbox_to_anchor=(0.5,1.1), loc="upper center", ncol=3)
            repImageBase64 = encodeRepImage('plt', plt, format='png')
        elif problem_type == "regression" and len(continuous_features) == 2 and len(categorical_features) == 0:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            xs = train_feature_data.iloc[:, 0].reset_index(drop=True)
            ys = train_feature_data.iloc[:, 1].reset_index(drop=True)
            zs = train_result_data.reset_index(drop=True)
            ax.scatter(xs, ys, zs, marker='o', color='#03bffe', label='Training Data')
            xs = feature_data.iloc[:, 0]
            ys = feature_data.iloc[:, 1]
            xs = numpy.linspace(int(numpy.floor(min(xs)))-1,int(numpy.ceil(max(xs)))+1,200)
            ys = numpy.linspace(int(numpy.floor(min(ys)))-1,int(numpy.ceil(max(ys)))+1,200)
            xs, ys = numpy.meshgrid( xs, ys )
            coords = numpy.vstack([xs.ravel(), ys.ravel()])
            coordsScaled = scale.transform( coords.transpose() )
            zs = model.predict( coordsScaled )
            zs = zs.reshape(xs.shape)
            surf = ax.plot_surface(xs, ys, zs, alpha = 0.5, cmap='copper', color="#03bffe", label='Model')
            #plot 3d testing data
            xs = test_feature_data.iloc[:, 0].reset_index(drop=True)
            ys = test_feature_data.iloc[:, 1].reset_index(drop=True)
            zs = test_result_data.reset_index(drop=True)
            ax.scatter(xs, ys, zs, marker='x', color='#ff845b', label='Testing Data')
            ax.set_xlabel(train_feature_data.columns[0])
            ax.set_ylabel(train_feature_data.columns[1])
            ax.set_zlabel(result)
            surf._edgecolors2d = surf._edgecolor3d
            surf._facecolors2d = surf._facecolor3d
            ax.legend(bbox_to_anchor=(0.5,1.1), loc="upper center", ncol=3)
            repImageBase64 = encodeRepImage('plt', plt, format='png')
        else:
            repImageCreated=False
    elif model_type == "LinReg":
        if len(features) == 1 and len(categorical_features) == 0:
            plt.scatter(train_feature_data, train_result_data, marker='o', color='#03bffe', label='Training Data')
            xs = numpy.linspace(numpy.floor(min(train_feature_data.iloc[:, 0])),numpy.ceil(max(train_feature_data.iloc[:, 0])),100)
            xsScaled = scale.transform(xs.reshape(-1,1))
            plt.xlabel(train_feature_data.columns[0])
            plt.ylabel(result)
            plt.plot( xs, model.coef_[0]*xsScaled + model.predict(numpy.array([[0]])), color="#fe4203", label='Model')
            plt.scatter(test_feature_data, test_result_data, marker='x', color='#ff845b', label='Testing Data')
            plt.legend(bbox_to_anchor=(0.5,1.1), loc="upper center", ncol=3)
            repImageBase64 = encodeRepImage('plt', plt, format='png')
        elif len(continuous_features) == 2 and len(categorical_features) == 0:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            xs = train_feature_data.iloc[:, 0]
            ys = train_feature_data.iloc[:, 1]
            zs = train_result_data
            ax.scatter(xs, ys, zs, marker='o', color='#03bffe', label='Training Data')
            xs, ys = numpy.mgrid[ range(int(numpy.floor(min(xs))),int(numpy.ceil(max(xs)))), range(int(numpy.floor(min(ys))),int(numpy.ceil(max(ys)))) ]
            coords = numpy.vstack([xs.ravel(), ys.ravel()])
            coordsScaled = scale.transform( coords.transpose() )
            xsScaled = coordsScaled[:,0]
            ysScaled = coordsScaled[:,1]
            xsScaled = xsScaled.reshape(len(xs),len(xs[0]))
            ysScaled = ysScaled.reshape(len(xs),len(xs[0]))
            zs = model.coef_[0]*xsScaled + model.coef_[1]*ysScaled + model.predict(numpy.array([[0, 0]]))
            surf = ax.plot_surface(xs, ys, zs, alpha = 0.5, cmap='copper', color="#03bffe", label='Model')
            #plot 3d testing data
            xs = test_feature_data.iloc[:, 0]
            ys = test_feature_data.iloc[:, 1]
            zs = test_result_data
            ax.scatter(xs, ys, zs, marker='x', color='#ff845b', label='Testing Data')
            ax.set_xlabel(train_feature_data.columns[0])
            ax.set_ylabel(train_feature_data.columns[1])
            ax.set_zlabel(result)
            surf._edgecolors2d = surf._edgecolor3d
            surf._facecolors2d = surf._facecolor3d
            ax.legend(bbox_to_anchor=(0.5,1.1), loc="upper center", ncol=3)
            repImageBase64 = encodeRepImage('plt', plt, format='png')
        else:
            repImageCreated=False
    elif model_type == "PolyFit":
        if len(features) == 1:
            plt.scatter(train_feature_data, train_result_data, marker='o', color='#03bffe', label='Training Data')
            xs = numpy.linspace(numpy.floor(min(train_feature_data.iloc[:, 0])),numpy.ceil(max(train_feature_data.iloc[:, 0])),100)
            xsScaled = scale.transform(xs.reshape(-1,1))
            plt.xlabel(train_feature_data.columns[0])
            plt.ylabel(result)
            plt.plot( xs, model(xsScaled), color="#fe4203", label='Model')
            plt.scatter(test_feature_data, test_result_data, marker='x', color='#ff845b', label='Testing Data')
            plt.legend(bbox_to_anchor=(0.5,1.1), loc="upper center", ncol=3)
            repImageBase64 = encodeRepImage('plt', plt, format='png')
        else:
            repImageCreated=False

    # Ensure plot object always clear for next request
    plt.clf()

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
        'repImageCreated': repImageCreated,
    }

    if (repImageBase64):
        outputs['repImageBase64'] = repImageBase64

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

def encodeRepImage(type, imageObject, **options):
    """
    Encodes a model representation image object to a base64 string.

    Args:
        imageObject: The image object (Matplotlib plot or graph).
        **options: Additional keyword arguments to pass to `savefig` if using plt.

    Returns:
        A base64-encoded string of the image.
    """
    repImageBuffer = io.BytesIO()

    if (type == 'plt'):
        imageObject.savefig(repImageBuffer, bbox_inches='tight', **options)
    elif (type == 'graph'):
        imageObject.write_png(repImageBuffer)

    repImageBuffer.seek(0)
    repImageBase64 = base64.b64encode(repImageBuffer.read()).decode('utf-8')
    repImageBuffer.close()

    return repImageBase64

def removeValuesFromNodes(graph):
    for node in graph.get_node_list():
        label = node.get('label')
        if label:
            # Assumes values are in "value = [...]" format, remove "value = [...]\\n"
            newLabel = re.sub(r"value = \[.*?\]\\n", "", label)
            node.set("label", newLabel)

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
