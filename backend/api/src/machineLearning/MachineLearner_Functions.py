from .SessionDataManager import SessionDataManager
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
from sklearn import linear_model
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score #CROSS VAL SCORE like many test/train splits and setting an expectation for accuracy on unseen data
from .custom_metric import mydist
import pandas
import pydotplus
from ordered_set import OrderedSet
from joblib import dump, load
import csv
import sys
import os
import colorsys
from contextlib import contextmanager
import io
import base64
import re
# import threading
# import _thread
# import multiprocessing

from api.rootDirectory import API_ROOT_DIRECTORY
DATASETS_DIRECTORY = os.path.join(API_ROOT_DIRECTORY, 'data')

session_data_manager = SessionDataManager()

def choose_dataset(session_id, filename, file):
    if filename.startswith("Examples/"):
        session_data_manager.add_dataset(session_id, {
            'filename': filename,
            'is_default_dataset': True
        })
    else:
        add_dataset_to_session_data(session_id, filename, file)

    dataset = get_dataset(session_id)

    # Identify and return the field names
    fields = dataset.columns.values.tolist()
    nonCtsFields = list(dataset.dtypes[ (dataset.dtypes != "int64") & (dataset.dtypes != "float64")].index)
    return {'fields': fields, 'nonCtsFields': nonCtsFields}

def get_dataset(session_id):
    """
    Return the dataset associated with the given session ID.

    If the dataset is one of the example datasets, it is read from disk.
    Otherwise, it is retrieved from the session data.

    Parameters
    ----------
    session_id : str
        Session ID to retrieve the dataset for.

    Returns
    -------
    dataset : pandas.DataFrame
        The dataset associated with the given session ID.
    """
    dataset = session_data_manager.get_session_data(session_id)['dataset']

    if dataset['is_default_dataset']:
        return pandas.read_csv(DATASETS_DIRECTORY+"/CO2 Emissions.csv")
    else:
        return dataset['dataset']

def add_dataset_to_session_data(session_id, filename, file):
    """
    Add a dataset to the session data.

    Parameters
    ----------
    session_id : str
        Session ID to add the dataset to.
    filename : str
        Name of the file to add.
    file : str
        The contents of the file to add.
    """
    
    # Prevent files that are too big from being uploaded
    if len(file.encode("utf8")) > 2000000:
        return None

    file_lines = [x  for x in file.split('\n')]
    file_entries = []
    for row in csv.reader(file_lines, delimiter=","):
        file_entries.append(row)

    dataset_as_dataframe = pandas.DataFrame(file_entries)
    dataset = {
        'filename': filename,
        'dataset': dataset_as_dataframe
    }

    session_data_manager.add_dataset(session_id, dataset)

# class TimeoutException(Exception):
#     def __init__(self, msg=''):
#         self.msg = msg

# class MyException(Exception):
#     pass

# @contextmanager
# def time_limit(cwd, seconds, msg=''):
# # def time_limit(supervision, problemType, methodML, polyDeg, ctsParams, cateParams, result, testProp, datasetName, cwd, seconds, msg=''):
#     # timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
#     timer = threading.Timer( seconds, lambda: (print("TIMED OUT SO CANCELLED"), MyException) )
#     timer.start()
#     print("TIMER STARTED AND IS")
#     try:
#         print("TRYING")
#         # yield
#         yield
#     except MyException:
#         # os.chdir(cwd)
#         # print(os.getcwd())
#         print("TIMED OUT SO CANCELLED")
#         raise TimeoutException("Timed out for operation {}".format(msg)) 
#     finally:
#         timer.cancel()
#         print("STOPPED?")

# def machineLearnerTimed(supervision, problemType, methodML, polyDeg, ctsParams, cateParams, result, testProp, datasetName):
#     cwd = os.getcwd()
#     print(cwd)
#     try:
#         print("KICKING OFF ML WITH TIME LIMIT")
#         with time_limit(cwd, 5, 'sleep'):
#         # return time_limit(supervision, problemType, methodML, polyDeg, ctsParams, cateParams, result, testProp, datasetName, cwd, 5, 'sleep')
#             return machineLearner(supervision, problemType, methodML, polyDeg, ctsParams, cateParams, result, testProp, datasetName)
#     except TimeoutException:
#         print("TIME RAN OUT EXCEPTION, RETURNING NOTHING")
#         return
#     print("HERE 4")


def machineLearner(supervision, problemType, methodML, polyDeg, ctsParams, cateParams, result, testProp, datasetName, sessionID): #, accTrain, accTest, inpVal):
    # Disable pandas warning "A value is trying to be set on a copy of a slice from a DataFrame"
    pandas.options.mode.chained_assignment = None  # default='warn'

    #Scale to use: StandardScaler(), PowerTransformer(,), ...
    scale = StandardScaler()
    #Method of learning: DT, LinReg, PolyFit (1d), ...  . polyDeg degress of polynomial if PolyFit selected
    #Features to analyse, independent X, and proportion of test vs train. Cts followed by categorical
    features = ctsParams + cateParams
    fieldsOfInterest = features + [result]
    global ncts
    global ncate
    global ntotal
    ncts = len(ctsParams)
    ncate = len(cateParams)
    ntotal = ncts+ncate
    #ctsParams = [feat for feat in features if feat not in cateParams]
    #Result of interest, dependent y, and is the result continuous or categorical?
    #Features which are categorical - to be converted to integers
    #Specific value of interest/to predict, 1 entry per feature

    dictionary = {}

    # print("AAAAAAAAAAAAAAAAAAAAAAAAA DATASET NAME IS: " + datasetName)
    # if problemType=="classification":
    #     strTypes = cateParams+[result]
    # else:
    #     strTypes = cateParams
    # # print(strTypes)
    # dict_dtypes = {x : 'str'  for x in strTypes}
    # dataSet = pandas.read_csv(nwd + '/' + datasetName, dtype=dict_dtypes)

    # Todo: Refactor this to get data from csv if default dataset, or from session_data_manager otherwise
    dataSet = pandas.read_csv(DATASETS_DIRECTORY+'/'+datasetName)
    dataSet = dataSet.dropna(axis=0, how='any', thresh=None, subset=fieldsOfInterest, inplace=False)

    if methodML in ['DT', 'KNN']:
        cateIndicators = False #False: converts each option to integers sequentially.
    else: #LinReg or PolyFit etc, need a cts result, and indicator dummy variables (unless categories are sequential, e.g. cold, medium, hot)
        cateIndicators = True  #True: Convert category parameters to (n-1) indicator dummy variables, 1 for each n options, minus 1 to drop the first column which is not independent of the rest.

    #Define training data, independent upper case X, dependent lower case y
    x = dataSet[features]
    dict_dtypes = {x : 'str'  for x in cateParams}
    x = x.astype(dtype = dict_dtypes, copy=True)
    y = dataSet[result]
    if problemType == "classification":
        y = y.astype(dtype = 'str', copy=True)
    dictionaryResult = dict()
    if problemType == "classification":
        value = 0
        for option in OrderedSet(y):
            dictionaryResult[option] = value
            value += 1
        y = y.map(dictionaryResult)
    catex = x[cateParams]
    ctsx = x[ctsParams]
    for category in cateParams:
        if cateIndicators == True:  # One Hot Encoding
            value = 0   #----------------- Dictionary not used here, just for input validation
            dictionary[category] = dict()
            for option in OrderedSet(catex[category]):
                dictionary[category][option] = value
                value += 1
            catex = pandas.get_dummies(data=catex, drop_first=True, columns = [category])
        else:
            value = 0
            dictionary[category] = dict()
            for option in OrderedSet(catex[category]):
                dictionary[category][option] = value
                value += 1
            catex[category] = catex[category].map(dictionary[category])
    
    testSize = round( min( max(2, testProp*len(x)), len(x)-2 ) ) # Need at least 2 test/train sample (for measuring regression accuracy). Borderline case.
    trainCtsx, testCtsx, trainCatex, testCatex, trainy, testy = train_test_split(ctsx, catex, y, test_size = testSize)
    cateCols = trainCatex.columns
    trainx = pandas.concat([trainCtsx, trainCatex], axis=1)
    testx = pandas.concat([testCtsx, testCatex], axis=1)
    if len(ctsParams) > 0: #Scaling and assembling the FinalX sets
        if methodML in ["LinReg", "PolyFit", "KNN"]: #Any model requiring scaling
            trainScaledX = scale.fit_transform(trainCtsx.values) #Scaling is defined here, by the training dataset
            trainScaledX = pandas.DataFrame(trainScaledX, columns=ctsParams)
            testScaledX = scale.transform(testCtsx.values)       #And applied as defined here, to the testing dataset
            testScaledX = pandas.DataFrame(testScaledX, columns=ctsParams)
            trainCatex = trainCatex.reset_index(drop=True)
            testCatex = testCatex.reset_index(drop=True)
            trainy = trainy.reset_index(drop=True)
            testy = testy.reset_index(drop=True)
            if len(cateParams) > 0:
                trainFinalX = pandas.concat([trainScaledX, trainCatex], axis=1).to_numpy()   #1: Cts params, scaling, cate params
                testFinalX = pandas.concat([testScaledX, testCatex], axis=1).to_numpy()      #1: Cts params, scaling, cate params
            else:
                trainFinalX = trainScaledX.to_numpy()                                                               #2: Cts params, scaling, no cate params
                testFinalX = testScaledX.to_numpy()                                                                 #2: Cts params, scaling, no cate params
        else: #No scaling required
            if len(cateParams) > 0:
                trainFinalX = pandas.concat([trainCtsx, trainCatex], axis=1)                                        #3: Cts params, no scaling, cate params
                testFinalX = pandas.concat([testCtsx, testCatex], axis=1)                                           #3: Cts params, no scaling, cate params
            else:
                trainFinalX = trainCtsx                                                                             #4: Cts params, no scaling, no cate params
                testFinalX = testCtsx                                                                               #4: Cts params, no scaling, no cate params
    else:
            if len(cateParams) > 0:
                trainFinalX = trainCatex                                                                            #5: No cts params, no scaling, cate params
                testFinalX = testCatex                                                                              #5: No cts params, no scaling, cate params
            else:
                # print('You have not selected any features')                                                       #6: No params
                raise SystemExit(0)

    #Create model (decision tree, linear regression etc)
    print("DATA SORTED AND SETS DEFINED, NOW FITTING MODEL")
    if methodML == 'DT':
        if problemType == "classification":
            decTree = DecisionTreeClassifier()#max_depth=7) #Add parameters for max depth, criterion etc
        elif problemType == "regression":
            decTree = DecisionTreeRegressor()#max_depth=7) #Add parameters for max depth, criterion etc
        decTree = decTree.fit(trainFinalX.values,trainy.values)
        session_data_manager.add_model(sessionID, decTree)
    elif methodML == "KNN": # USES STANDARD SCALING ON CTS FEATURES - to get closer to a Gaussian distribution        
        if problemType == "classification":
            if ncts == ntotal:
                KNN = KNeighborsClassifier(n_neighbors=min(4, len(trainFinalX)), weights="distance", metric='minkowski')
            elif ncate == ntotal:
                KNN = KNeighborsClassifier(n_neighbors=min(4, len(trainFinalX)), weights="distance", metric='hamming')
            else:
                KNN = KNeighborsClassifier(n_neighbors=min(4, len(trainFinalX)), weights="distance", metric=mydist, metric_params={"ncts": ncts, "ncate": ncate}, algorithm='ball_tree', leaf_size=2)
        elif problemType == "regression":
            if ncts == ntotal:
                KNN = KNeighborsRegressor(n_neighbors=min(4, len(trainFinalX)), weights="distance", metric='minkowski')
            elif ncate == ntotal:
                KNN = KNeighborsRegressor(n_neighbors=min(4, len(trainFinalX)), weights="distance", metric='hamming')
            else:
                KNN = KNeighborsRegressor(n_neighbors=min(4, len(trainFinalX)), weights="distance", metric=mydist, metric_params={"ncts": ncts, "ncate": ncate}, algorithm='ball_tree', leaf_size=2)
        KNN.fit(trainFinalX,trainy)
        session_data_manager.add_model(sessionID, KNN)
    elif methodML == "LinReg":
        regr = linear_model.LinearRegression()
        if len(ctsParams) == 0:
            trainFinalX = trainFinalX.to_numpy()
            testFinalX = testFinalX.to_numpy()
            regr.fit(trainFinalX, trainy)
            session_data_manager.add_model(sessionID, regr)
        else:
            regr.fit(trainFinalX, trainy)
            session_data_manager.add_model(sessionID, regr)
    elif methodML == "PolyFit":
        polyModel1d = numpy.poly1d(numpy.polyfit(trainFinalX.ravel(), trainy, polyDeg))
        session_data_manager.add_model(sessionID, polyModel1d)

    #Measure fit of model
    print("MODEL FITTED AND SAVED, MEASURING ACCURACY OF MODEL")
    if methodML == 'DT': #Look at "accuracy_score", based on matching result labels
        if problemType == "classification":
            predictedTrain = decTree.predict(trainFinalX.values)
            predictedTest = decTree.predict(testFinalX.values)
            accuracyTrain = accuracy_score(trainy, predictedTrain)
            accuracyTest = accuracy_score(testy, predictedTest)
            macroPrecTrain = precision_score(trainy, predictedTrain, average="macro") # AVERAGE OF INDIV CLASS PRECISIONS = numpy.average( TP / (TP+FP) ), good for spotting poor precision on minority classes
            microPrecTrain = precision_score(trainy, predictedTrain, average="micro") # WEIGHTED AVERAGE OF INDIV CLASS PRECISIONS = TP.sum() / (TP.sum()+FP.sum()), good if you only care about precision in majority of cases
            macroRecallTrain = recall_score(trainy, predictedTrain, average="macro")
            microRecallTrain = recall_score(trainy, predictedTrain, average="micro") # Micro Precision/Micro Recall/Micro F1-score, all equal to Accuracy when items can only have one label each (i.e. non-Multi-Label problem)
            macroPrecTest = precision_score(testy, predictedTest, average="macro")
            microPrecTest = precision_score(testy, predictedTest, average="micro")
            macroRecallTest = recall_score(testy, predictedTest, average="macro")
            microRecallTest = recall_score(testy, predictedTest, average="micro")
        elif problemType == "regression":
            accuracyTrain = r2_score(trainy, decTree.predict(trainFinalX.values)) # Coefficient of Determination
            accuracyTest = r2_score(testy, decTree.predict(testFinalX.values))
    elif methodML == "KNN":
        if problemType == "classification":
            predictedTrain = KNN.predict( numpy.array(trainFinalX) )
            predictedTest = KNN.predict( numpy.array(testFinalX) )
            accuracyTrain = accuracy_score(trainy, predictedTrain)
            accuracyTest = accuracy_score(testy, predictedTest)
            macroPrecTrain = precision_score(trainy, predictedTrain, average="macro")
            microPrecTrain = precision_score(trainy, predictedTrain, average="micro")
            macroRecallTrain = recall_score(trainy, predictedTrain, average="macro")
            microRecallTrain = recall_score(trainy, predictedTrain, average="micro")
            macroPrecTest = precision_score(testy, predictedTest, average="macro")
            microPrecTest = precision_score(testy, predictedTest, average="micro")
            macroRecallTest = recall_score(testy, predictedTest, average="macro")
            microRecallTest = recall_score(testy, predictedTest, average="micro")
        elif problemType =="regression":
            accuracyTrain = r2_score(trainy, KNN.predict( numpy.array(trainFinalX) ) ) # Coefficient of Determination
            accuracyTest = r2_score(testy, KNN.predict( numpy.array(testFinalX) ) )
    elif methodML == "LinReg":
        # r-squared, measure of how well model fits data (Coefficient of Determination, greater is better fit). Ideally R^2 will be similar for the training and testing data sets. When R2<0, a horizontal line (mean) explains the data better than your model.
        accuracyTrain = r2_score(trainy, regr.predict( numpy.array(trainFinalX) ))
        accuracyTest = r2_score(testy, regr.predict( numpy.array(testFinalX) ))
    elif methodML == "PolyFit":
        accuracyTrain = r2_score(trainy, polyModel1d( numpy.array(trainFinalX) ))
        accuracyTest = r2_score(testy, polyModel1d( numpy.array(testFinalX) ))

    # Default initialization to avoid checking for non-existence later
    repImageBase64 = None

    # Draw/plot the model tree/3d training data
    repImageCreated=True
    if methodML == "DT":
        if problemType == "classification":
            if len(ctsParams) == 2 and len(cateParams) == 0:
                trainy = trainy.reset_index(drop=True)
                testy = testy.reset_index(drop=True)
                xsTest = testx.iloc[:, 0].reset_index(drop=True)
                ysTest = testx.iloc[:, 1].reset_index(drop=True)
                xsTrain = trainx.iloc[:, 0].reset_index(drop=True)
                ysTrain = trainx.iloc[:, 1].reset_index(drop=True)
                xs = x.iloc[:, 0]
                ys = x.iloc[:, 1]
                xs = numpy.linspace(int(numpy.floor(min(xs)))-1,int(numpy.ceil(max(xs)))+1,200)
                ys = numpy.linspace(int(numpy.floor(min(ys)))-1,int(numpy.ceil(max(ys)))+1,200)
                xs, ys = numpy.meshgrid( xs, ys )
                coords = numpy.vstack([xs.ravel(), ys.ravel()]).transpose()
                zs = decTree.predict(coords)
                lightcmap = cm.get_cmap('rainbow', len(dictionaryResult))
                darkcmap = cm.get_cmap('rainbow', len(dictionaryResult))
                lightcolours=[]
                listOfzs = list(set(zs))
                listOfzs.sort()
                for ind in range(len(dictionaryResult)):
                    c = colorsys.rgb_to_hls(lightcmap(ind)[0], lightcmap(ind)[1], lightcmap(ind)[2])
                    lightcolours.append( tuple(colorsys.hls_to_rgb(c[0], 1 - 0.5 * (1 - c[1]), c[2]*0.7)) )
                lightcmap = cm.colors.ListedColormap(lightcolours)
                zs = zs.reshape(xs.shape)
                plt.pcolormesh(xs, ys, zs, cmap=lightcmap, vmin=min(dictionaryResult.values()), vmax=max(dictionaryResult.values()))
                count = 0
                for res in dictionaryResult:
                    count += 1
                    dictRes = dictionaryResult[res]
                    indices = numpy.where(trainy == dictRes)[0]
                    plt.scatter(xsTrain[indices], ysTrain[indices], color=darkcmap(dictRes), marker='o', label=(count>18)*"_"+res[0:8])
                    indices = numpy.where(testy == dictRes)[0]
                    plt.scatter(xsTest[indices], ysTest[indices], color=darkcmap(dictRes), marker='x')
                plt.xlabel(trainx.columns[0])
                plt.ylabel(trainx.columns[1])
                plt.legend(bbox_to_anchor=(1,1), loc="upper left")
                repImageBase64 = encodeRepImage('plt', plt, format='png')
            else:
                data = tree.export_graphviz(decTree, out_file=None, feature_names=features, class_names=list(dictionaryResult), max_depth=4, filled=True)
                graph = pydotplus.graph_from_dot_data(data)
                removeValuesFromNodes(graph)
                repImageBase64 = encodeRepImage('graph', graph)
        elif problemType == "regression":
            if len(ctsParams) == 1 and len(cateParams) == 0:
                plt.scatter(trainx, trainy, marker='o', color='#03bffe', label='Training Data')
                xs = x.iloc[:, 0]
                xs = numpy.linspace(int(numpy.floor(min(xs)))-1,int(numpy.ceil(max(xs)))+1,200).reshape(-1,1)
                zs = decTree.predict(xs)
                plt.xlabel(trainx.columns[0])
                plt.ylabel(result)
                plt.plot( xs, zs, color="#fe4203", label='Model')
                plt.scatter(testx, testy, marker='x', color='#ff845b', label='Testing Data')
                plt.legend(bbox_to_anchor=(0.5,1.1), loc="upper center", ncol=3)
                repImageBase64 = encodeRepImage('plt', plt, format='png')
            elif len(ctsParams) == 2 and len(cateParams) == 0:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                xs = trainx.iloc[:, 0].reset_index(drop=True)
                ys = trainx.iloc[:, 1].reset_index(drop=True)
                zs = trainy.reset_index(drop=True)
                ax.scatter(xs, ys, zs, marker='o', color='#03bffe', label='Training Data')
                xs = x.iloc[:, 0]
                ys = x.iloc[:, 1]
                xs = numpy.linspace(int(numpy.floor(min(xs)))-1,int(numpy.ceil(max(xs)))+1,200)
                ys = numpy.linspace(int(numpy.floor(min(ys)))-1,int(numpy.ceil(max(ys)))+1,200)
                xs, ys = numpy.meshgrid( xs, ys )
                coords = numpy.vstack([xs.ravel(), ys.ravel()]).transpose()
                zs = decTree.predict(coords)
                zs = zs.reshape(xs.shape)
                surf = ax.plot_surface(xs, ys, zs, alpha = 0.5, cmap='copper', color="#03bffe", label='Model')
                #plot 3d testing data
                xs = testx.iloc[:, 0].reset_index(drop=True)
                ys = testx.iloc[:, 1].reset_index(drop=True)
                zs = testy.reset_index(drop=True)
                ax.scatter(xs, ys, zs, marker='x', color='#ff845b', label='Testing Data')
                ax.set_xlabel(trainx.columns[0])
                ax.set_ylabel(trainx.columns[1])
                ax.set_zlabel(result)
                surf._edgecolors2d = surf._edgecolor3d
                surf._facecolors2d = surf._facecolor3d
                ax.legend(bbox_to_anchor=(0.5,1.1), loc="upper center", ncol=3)
                repImageBase64 = encodeRepImage('plt', plt, format='png')
            else:
                data = tree.export_graphviz(decTree, out_file=None, feature_names=features, max_depth=4, filled=True)
                graph = pydotplus.graph_from_dot_data(data)
                removeValuesFromNodes(graph)
                repImageBase64 = encodeRepImage('graph', graph)
        else:
            repImageCreated=False
    elif methodML == "KNN":
        if problemType == "classification" and len(ctsParams) == 2 and len(cateParams) == 0:
            trainy = trainy.reset_index(drop=True)
            testy = testy.reset_index(drop=True)
            xsTest = testx.iloc[:, 0].reset_index(drop=True)
            ysTest = testx.iloc[:, 1].reset_index(drop=True)
            xsTrain = trainx.iloc[:, 0].reset_index(drop=True)
            ysTrain = trainx.iloc[:, 1].reset_index(drop=True)
            xs = x.iloc[:, 0]
            ys = x.iloc[:, 1]
            xs = numpy.linspace(int(numpy.floor(min(xs)))-1,int(numpy.ceil(max(xs)))+1,200)
            ys = numpy.linspace(int(numpy.floor(min(ys)))-1,int(numpy.ceil(max(ys)))+1,200)
            xs, ys = numpy.meshgrid( xs, ys )
            coords = numpy.vstack([xs.ravel(), ys.ravel()])
            coordsScaled = scale.transform( coords.transpose() )
            zs = KNN.predict( coordsScaled )
            lightcmap = cm.get_cmap('rainbow', len(dictionaryResult))
            darkcmap = cm.get_cmap('rainbow', len(dictionaryResult))
            lightcolours=[]
            listOfzs = list(set(zs))
            listOfzs.sort()
            for ind in range(len(dictionaryResult)):
                c = colorsys.rgb_to_hls(lightcmap(ind)[0], lightcmap(ind)[1], lightcmap(ind)[2])
                lightcolours.append( tuple(colorsys.hls_to_rgb(c[0], 1 - 0.5 * (1 - c[1]), c[2]*0.7)) )
            lightcmap = cm.colors.ListedColormap(lightcolours)
            zs = zs.reshape(xs.shape)
            plt.pcolormesh(xs, ys, zs, cmap=lightcmap, vmin=min(dictionaryResult.values()), vmax=max(dictionaryResult.values()))
            count = 0
            for res in dictionaryResult:
                count += 1
                dictRes = dictionaryResult[res]
                indices = numpy.where(trainy == dictRes)[0]
                plt.scatter(xsTrain[indices], ysTrain[indices], color=darkcmap(dictRes), marker='o', label=(count>18)*"_"+res[0:8])
                indices = numpy.where(testy == dictRes)[0]
                plt.scatter(xsTest[indices], ysTest[indices], color=darkcmap(dictRes), marker='x')
            plt.xlabel(trainx.columns[0])
            plt.ylabel(trainx.columns[1])
            plt.legend(bbox_to_anchor=(1,1), loc="upper left")
            repImageBase64 = encodeRepImage('plt', plt, format='png')
        elif problemType == "regression" and len(ctsParams) == 1 and len(cateParams) == 0:
            plt.scatter(trainx, trainy, marker='o', color='#03bffe', label='Training Data')
            xs = x.iloc[:, 0]
            xs = numpy.linspace(int(numpy.floor(min(xs)))-1,int(numpy.ceil(max(xs)))+1,200)
            xsScaled = scale.transform(xs.reshape(-1,1))
            zs = KNN.predict( xsScaled )
            plt.xlabel(trainx.columns[0])
            plt.ylabel(result)
            plt.plot( xs, zs, color="#fe4203", label='Model')
            plt.scatter(testx, testy, marker='x', color='#ff845b', label='Testing Data')
            plt.legend(bbox_to_anchor=(0.5,1.1), loc="upper center", ncol=3)
            repImageBase64 = encodeRepImage('plt', plt, format='png')
        elif problemType == "regression" and len(ctsParams) == 2 and len(cateParams) == 0:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            xs = trainx.iloc[:, 0].reset_index(drop=True)
            ys = trainx.iloc[:, 1].reset_index(drop=True)
            zs = trainy.reset_index(drop=True)
            ax.scatter(xs, ys, zs, marker='o', color='#03bffe', label='Training Data')
            xs = x.iloc[:, 0]
            ys = x.iloc[:, 1]
            xs = numpy.linspace(int(numpy.floor(min(xs)))-1,int(numpy.ceil(max(xs)))+1,200)
            ys = numpy.linspace(int(numpy.floor(min(ys)))-1,int(numpy.ceil(max(ys)))+1,200)
            xs, ys = numpy.meshgrid( xs, ys )
            coords = numpy.vstack([xs.ravel(), ys.ravel()])
            coordsScaled = scale.transform( coords.transpose() )
            zs = KNN.predict( coordsScaled )
            zs = zs.reshape(xs.shape)
            surf = ax.plot_surface(xs, ys, zs, alpha = 0.5, cmap='copper', color="#03bffe", label='Model')
            #plot 3d testing data
            xs = testx.iloc[:, 0].reset_index(drop=True)
            ys = testx.iloc[:, 1].reset_index(drop=True)
            zs = testy.reset_index(drop=True)
            ax.scatter(xs, ys, zs, marker='x', color='#ff845b', label='Testing Data')
            ax.set_xlabel(trainx.columns[0])
            ax.set_ylabel(trainx.columns[1])
            ax.set_zlabel(result)
            surf._edgecolors2d = surf._edgecolor3d
            surf._facecolors2d = surf._facecolor3d
            ax.legend(bbox_to_anchor=(0.5,1.1), loc="upper center", ncol=3)
            repImageBase64 = encodeRepImage('plt', plt, format='png')
        else:
            repImageCreated=False
    elif methodML == "LinReg":
        if len(features) == 1 and len(cateParams) == 0:
            plt.scatter(trainx, trainy, marker='o', color='#03bffe', label='Training Data')
            xs = numpy.linspace(numpy.floor(min(trainx.iloc[:, 0])),numpy.ceil(max(trainx.iloc[:, 0])),100)
            xsScaled = scale.transform(xs.reshape(-1,1))
            plt.xlabel(trainx.columns[0])
            plt.ylabel(result)
            plt.plot( xs, regr.coef_[0]*xsScaled + regr.predict(numpy.array([[0]])), color="#fe4203", label='Model')
            plt.scatter(testx, testy, marker='x', color='#ff845b', label='Testing Data')
            plt.legend(bbox_to_anchor=(0.5,1.1), loc="upper center", ncol=3)
            repImageBase64 = encodeRepImage('plt', plt, format='png')
        elif len(ctsParams) == 2 and len(cateParams) == 0:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            xs = trainx.iloc[:, 0]
            ys = trainx.iloc[:, 1]
            zs = trainy
            ax.scatter(xs, ys, zs, marker='o', color='#03bffe', label='Training Data')
            xs, ys = numpy.mgrid[ range(int(numpy.floor(min(xs))),int(numpy.ceil(max(xs)))), range(int(numpy.floor(min(ys))),int(numpy.ceil(max(ys)))) ]
            coords = numpy.vstack([xs.ravel(), ys.ravel()])
            coordsScaled = scale.transform( coords.transpose() )
            xsScaled = coordsScaled[:,0]
            ysScaled = coordsScaled[:,1]
            xsScaled = xsScaled.reshape(len(xs),len(xs[0]))
            ysScaled = ysScaled.reshape(len(xs),len(xs[0]))
            zs = regr.coef_[0]*xsScaled + regr.coef_[1]*ysScaled + regr.predict(numpy.array([[0, 0]]))
            surf = ax.plot_surface(xs, ys, zs, alpha = 0.5, cmap='copper', color="#03bffe", label='Model')
            #plot 3d testing data
            xs = testx.iloc[:, 0]
            ys = testx.iloc[:, 1]
            zs = testy
            ax.scatter(xs, ys, zs, marker='x', color='#ff845b', label='Testing Data')
            ax.set_xlabel(trainx.columns[0])
            ax.set_ylabel(trainx.columns[1])
            ax.set_zlabel(result)
            surf._edgecolors2d = surf._edgecolor3d
            surf._facecolors2d = surf._facecolor3d
            ax.legend(bbox_to_anchor=(0.5,1.1), loc="upper center", ncol=3)
            repImageBase64 = encodeRepImage('plt', plt, format='png')
        else:
            repImageCreated=False
    elif methodML == "PolyFit":
        if len(features) == 1:
            plt.scatter(trainx, trainy, marker='o', color='#03bffe', label='Training Data')
            xs = numpy.linspace(numpy.floor(min(trainx.iloc[:, 0])),numpy.ceil(max(trainx.iloc[:, 0])),100)
            xsScaled = scale.transform(xs.reshape(-1,1))
            plt.xlabel(trainx.columns[0])
            plt.ylabel(result)
            plt.plot( xs, polyModel1d(xsScaled), color="#fe4203", label='Model')
            plt.scatter(testx, testy, marker='x', color='#ff845b', label='Testing Data')
            plt.legend(bbox_to_anchor=(0.5,1.1), loc="upper center", ncol=3)
            repImageBase64 = encodeRepImage('plt', plt, format='png')
        else:
            repImageCreated=False

    # Ensure plot object always clear for next request
    plt.clf()

    print("REPRESENTATION CREATED AND SAVED, NOW SAVING MODEL SETTINGS/PARAMS")
    # os.chdir('../../../src/ModelData/Workings')
    # print(os.getcwd())
    # print(os.listdir(os.getcwd()))
    # os.chdir('./src/ModelData/Workings')
    # print("DIRECTORY AFTER EDITING REPRESENTATION")
    # print(os.getcwd())
    modelSettings = {
        "probType": problemType,
        "methodML": methodML,
        "features": features,
        "ctsParams": ctsParams,
        "cateParams": cateParams,
        "cateDict": dictionary,
        "resDict": dictionaryResult,
        "dataSet": dataSet,
        "scaler": scale
    }
    
    session_data_manager.add_model_settings(sessionID, modelSettings)

    print("SETTINGS SAVED, NOW SETTING UP INPUT VALIDATION AND EXPORTING")

    inputValidation = []

    for i in range(0, len(cateParams)):
        inputValidation.append(list(dictionary[cateParams[i]].keys()))

    # accTrain.value = accuracyTrain
    # accTest.value = accuracyTest
    # inpVal = inputValidation

    if problemType == "classification":
        mlOuts = {"accuracyTrain": accuracyTrain,
                "accuracyTest": accuracyTest,
                "precsRecs": {"macroPrecTrain": macroPrecTrain,
                                "microPrecTrain": microPrecTrain,
                                "macroRecallTrain": macroRecallTrain,
                                "microRecallTrain": microRecallTrain,
                                "macroPrecTest": macroPrecTest,
                                "microPrecTest": microPrecTest,
                                "macroRecallTest": macroRecallTest,
                                "microRecallTest": microRecallTest
                },
                "repImageCreated": repImageCreated
        }
    else:
        mlOuts = {"accuracyTrain": accuracyTrain,
                "accuracyTest": accuracyTest,
                "precsRecs": "",
                "repImageCreated": repImageCreated
        }

    if (repImageBase64):
        mlOuts['repImageBase64'] = repImageBase64

    return {"mlOuts": mlOuts, "inputValidation": inputValidation}

# def machineLearnerTimed(supervision, problemType, methodML, polyDeg, ctsParams, cateParams, result, testProp, datasetName):
#     cwd = os.getcwd()
#     print(cwd)

#     accTrain = multiprocessing.Value('d')
#     accTest = multiprocessing.Value('d')
#     inpVal = multiprocessing.Array('u', [])
    
#     print("STARTING TIMED FIT THREAD")
#     # p = threading.Thread(target = machineLearner, args = [supervision, problemType, methodML, polyDeg, ctsParams, cateParams, result, testProp, datasetName])
#     p = multiprocessing.Process(target = machineLearner, args = [supervision, problemType, methodML, polyDeg, ctsParams, cateParams, result, testProp, datasetName, accTrain, accTest, inpVal])
#     p.start()
#     p.join(5)

#     if p.is_alive():
#         print("Terminating fit thread")
#         os.chdir(cwd)
#         print(os.getcwd())
#         p.terminate()
#         p.join()
#         return
#     else:
#         print(os.getcwd())
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

    graph.write_png("updated_tree.png")

#Evaluate model at some particular value
def modelPrediction(predictAt, sessionID):
    #print("WWWWWWWWWW PREDICT START DIRECTORY")
    # os.chdir(sys.path[0]+'/..')
    #print(os.getcwd())

    # os.chdir('./src/ModelData/Workings')

    modelData = session_data_manager.get_session_data(sessionID)

    if modelData is None:
        # Handle the case where the model isn't found
        print(f"Model with session ID {sessionID} not found.")
        # Raise an exception, return an error response, or take other appropriate action
        raise ValueError(f"Model with session ID {sessionID} does not exist.")

    model = modelData['model']
    modelSettings = modelData['settings']
    # modelSettings = load("settings"+str(sessionID)+".joblib")

    problemType = modelSettings["probType"]
    methodML = modelSettings["methodML"]
    features = modelSettings["features"]
    ctsParams = modelSettings["ctsParams"]
    cateParams = modelSettings["cateParams"]
    dictionary = modelSettings["cateDict"]
    dictionaryResult = modelSettings["resDict"]
    dataSet = modelSettings["dataSet"]
    scale = modelSettings["scaler"]

    # model = load("model"+str(sessionID)+".joblib")

    # print("AAAAAAAAAAAAAAAAAA METHOD IS: " + methodML)
    # print("PREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEDICT AT IS:")
    # print(predictAt)

    if methodML == 'DT':
        predictAtNumerical = predictAt.copy()
        for i in range(0,len(features)):
            if features[i] in cateParams:
                category = features[i]
                # print("WWWWWWWWWWWWWWWWWWWWWWWWWW TRYING TO CONVERT: ")
                # print(category)
                # print(predictAtNumerical[i])
                # print(dictionary)
                predictAtNumerical[i] = dictionary[category][predictAtNumerical[i]]
        predicted = model.predict([predictAtNumerical])
        if problemType == "classification":
            predicted = [list(dictionaryResult.keys())[predicted[0]]]
    elif methodML == "KNN":
        predictAtAdj = []
        if len(ctsParams) > 0:
            predictAtAdj[0:len(ctsParams)] = scale.transform([predictAt[0:len(ctsParams)]]).ravel()
        if len(cateParams) > 0:
            for i in range(0,len(cateParams)):
                category = features[len(ctsParams) + i]
                predictAtAdj.append( dictionary[category][predictAt[len(ctsParams) + i]] )
        predicted = model.predict(numpy.array(predictAtAdj).reshape(1, -1))
        if problemType == "classification":
            predicted = [list(dictionaryResult.keys())[predicted[0]]]
    elif methodML == "LinReg":
        scaled = []
        predictCates = []
        if len(ctsParams) > 0:
            scaled = scale.transform([predictAt[0:len(ctsParams)]])
            scaled = scaled.ravel()
        if len(cateParams) > 0:
            for i in range(0,len(cateParams)):
                dummyCate = [0] * ( len(set(dataSet[cateParams[i]])) - 1 )
                onePos = sorted(set(dataSet[cateParams[i]])).index(predictAt[len(ctsParams)+i]) - 1
                if onePos > -1:
                    dummyCate[onePos] = 1
                predictCates = predictCates + dummyCate
        predictAtAdj = [*scaled, *predictCates]
        predicted = model.predict([predictAtAdj])
    elif methodML == "PolyFit":
        scaled = scale.transform([predictAt])
        predicted = model(scaled)[0]
        # print("AAAAAAAAAAAAAAAAAAAAAAA RESULT IS: ")
        # print(predicted)

    # os.chdir(sys.path[0]+'/..')
    
    #POSSIBLY ROUND Predicted 
    #round_to_n = lambda x, n: round(x, -int(floor(log10(x))) + (n - 1))
    
    # print("WWWWWWWWWW PREDICT FINAL DIRECTORY")
    # print(os.getcwd())
    return {"predictAt": predictAt, "prediction": predicted[0]}


# Ensure this is called when the website is closed or page refreshed
def clearModel(sessionID):
    session_data_manager.remove_session_data(sessionID)
    return
