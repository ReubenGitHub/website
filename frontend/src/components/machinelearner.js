import './pagestyles.css';
import {FormDataset, FormDefineModel, FormPredictAt, FormModelOutputs, FormModelPrediction} from './forms';
import React, {useState, useEffect, useCallback} from 'react';

export function MLerPage(props) {
    const [instTab, setInstTab] = useState(["active"]);
    const [sessionID, setSessionID] = useState(0);
    const [inputs, setInputs] = useState();
    const [mlOuts, setMlOuts] = useState(0);
    const [predictAt, setPredictAt] = useState([]);
    const [prediction, setPrediction] = useState("");
    const [predictionTitle, setPredictionTitle] = useState(false)
    const [predictFlag, setPredictFlag] = useState(false)
    const [datasetName, setDatasetName] = useState("");
    // const [datasetFields, setDatasetFields] = useState();
    const [datasetFields, setDatasetFields] = useState({fields: '', nonCtsFields: ''});
    const [datasetFeatures, setDatasetFeatures] = useState();
    const [datasetResultParam, setDatasetResultParam] = useState();
    const [loadingModelFit, setLoadingModelFit] = useState(false);
    const [loadingModelPredict, setLoadingModelPredict] = useState(false);
    const [loadingFields, setLoadingFields] = useState(false);
    const [noOfCtsParams, setNoOfCtsParams] = useState(0);
    const [inputValidation, setInputValidation] = useState({features: '', noOfCts: 0, options: [["Option"]]});

    const callbackFunction = (formsData) => {
        setInputs(formsData);
        setPredictAt([]);
        setPrediction("");
    }    
    const callbackFunctionDataset = (formsData) => {
        // console.log("DATASET CALLBKAC TRIGGETRED");
        if ( !(datasetName==formsData[0]) || !(arrayEquals(datasetFields['fields'],formsData[1]['fields'])) ) {
            setDatasetName(formsData[0]);
            setDatasetFields(formsData[1]);
            setDatasetFeatures();
            setMlOuts(0);
            setPredictionTitle(false);
        }
    }
    const callbackFunctionPredict = (formsData) => {
        setPredictAt(formsData); //Won't trigger useEffects if the underlying reference of the array items doesn't change
        setPredictFlag(!predictFlag);
    }
    
    //Set session ID only once, on initial loading
    useEffect(() => {
        setSessionID(Date.now());
    }, []);

    //Update model outputs and output-loading status upon trigger of inputs changing
    useEffect(() => {
        if (datasetName) {
            setLoadingModelFit(true);
            fetch('/api/mlModelFit', {
                method: 'post',
                headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    supervision: inputs[0],
                    problemtype: inputs[1],
                    mlmethod: inputs[2],
                    polydeg: inputs[3],
                    ctsparams: inputs[4],
                    cateparams: inputs[5],
                    resultparam: inputs[6],
                    testprop: inputs[7],
                    datasetname: datasetName,
                    sessionid: sessionID
                })
            }).then(res => res.json())
                .then(data => {setMlOuts(data.mlModelOutputs['mlOuts']);
                    setInputValidation({
                        features: inputs[4].concat(inputs[5]),
                        noOfCts: inputs[4].length,
                        options: data.mlModelOutputs['inputValidation']
                    });
                    setDatasetResultParam(inputs[6]);
                    setDatasetFeatures( inputs[4].concat(inputs[5]) );
                    setLoadingModelFit(false);
                    setPredictionTitle(true);
            });
        }
    }, [inputs]);

    //Update model prediction and prediction-loading status upon trigger of predictAt changing
    useEffect(() => {
        if (predictionTitle) {
            // console.log("PREDICTION API FUNCTION TRIGGERED");
            setLoadingModelPredict(true);
            fetch('/api/ml/predict', {
                method: 'post',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    predictAt,
                    sessionId: sessionID
                })
            })
                .then(res => res.json())
                .then(data => {
                    setPrediction(data.prediction)
                    setLoadingModelPredict(false)
                })
        }
    }, [predictFlag])

    //clear the model plot on unload, so they aren't presented with a random map image
    window.onbeforeunload = () => {
        // console.log("SCRIPT FOR UNLOAD ALREADY STARTED")
        // console.log("SCRIPT FOR UNLOAD PRE_FETCH FINISHED")
        fetch('/api/clearSessionData', {
            method: 'post',
            headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                sessionid: sessionID
            })
        })
        return ''; // Legacy method for cross browser support
      };

    //ML page, i.e. Parent, receives form inputs from Model Definition form, i.e. Child 1
    //Then hook is triggered by the state change in the inputs variable, which executes the python function via the api
    //Which updates the results, e.g. accuracy of the model, image...
    //These results are then passes from Parent to Output Forms, i.e. Child 2, for display
    return (
    <html>
    <body>

    <div class="about-section">
            <h1>Machine Learner</h1>
        </div>

        <div class="column-container">
            <div class="card">
                <div class="container-header-bar">
                    <a class={instTab[0]} onClick={() => setInstTab(["active"]) }>Getting Started</a>
                    <a class={instTab[1]} onClick={() => setInstTab( Array(2).fill("").fill("active",-1) ) }>Model Definition</a>
                    <a class={instTab[2]} onClick={() => setInstTab( Array(3).fill("").fill("active",-1) ) }>Features & Result</a>
                    <a class={instTab[3]} onClick={() => setInstTab( Array(4).fill("").fill("active",-1) ) }>Model Representation</a>
                    <a class={instTab[4]} onClick={() => setInstTab( Array(5).fill("").fill("active",-1) ) }>Accuracy</a>
                    <a class={instTab[5]} onClick={() => setInstTab( Array(6).fill("").fill("active",-1) ) }>Prediction</a>
                </div>
                <div class="container">
                    { (instTab[0]==="active") ?
                            <p>
                                This app is a general machine learner: designed to be intuitive, and built to allow you to fit a machine-led model to any data you like!
                                <br></br>
                                <br></br>
                                The stages involved in using this app are: choosing a dataset, selecting model parameters, selecting features and the result to predict, reviewing the model representation and accuracy, and using your model to predict results.
                                <br></br>
                                <br></br>
                                Click through these tabs for instructions to help you through each stage of building your models. To get started, read on below...
                                <br></br>
                                <br></br>
                                <table border="0">
                                    <tr>
                                        <td align="right" style={{"vertical-align": "top", "white-space": "nowrap"}}>
                                            <b>Data Selection</b>&nbsp;
                                        </td>
                                        <td align="left">
                                            Upload any csv dataset, where rows are entries, columns are fields, and column headers are included.
                                            <br></br>
                                            <br></br>
                                            Alternatively, select the default dataset, which contains vehicle emissions data such as model, engine size, and CO2 emissions (g/km).<br></br>
                                            The raw data was obtained from the UK government's page <i>carfueldata.vehicle-certification-agency.gov.uk/downloads/default.aspx</i>.<br></br>
                                            I then cleaned this data as part of my ML investigation into CO2 emissions <i>github.com/ReubenGitHub/ML-Vehicle-Emissions</i>.
                                        </td>
                                    </tr>
                                </table>
                            </p>
                        : (instTab[1]==="active") ?
                            <p>
                                <table border="0">
                                    <tr>
                                        <td align="right" style={{"vertical-align": "top", "white-space": "nowrap"}}>
                                            <b>Supervision</b>&nbsp;
                                        </td>
                                        <td align="left">
                                            Currently, all models on offer are <i>Supervised</i> (meaning the dataset contains a known result for every entry).
                                            <br></br><br></br>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td align="right" style={{"vertical-align": "top", "white-space": "nowrap"}}>
                                            <b>Problem Type</b>&nbsp;
                                        </td>
                                        <td align="left">
                                            Select <i>Regression</i> if the result you want to predict is a continuous variable, i.e. numerical and where some values can be "bigger" than others.
                                            <br></br><br></br>
                                            Select <i>Categorical</i> if the result you want to predict is a categorical variable, i.e. categories or types of something, with no sense of "bigger" or "smaller".
                                            <br></br><br></br>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td align="right" style={{"vertical-align": "top", "white-space": "nowrap"}}>
                                            <b>Machine Learning Method</b>&nbsp;
                                        </td>
                                        <td align="left">
                                            <i>Decision Trees</i> will derive a prediction through a sequence of decisions based on known features, similar to the game '20Q'.
                                            <br></br><br></br>
                                            <i>K-Nearest Neighbours</i> predicts a result by averaging the results of K training samples which have feature values closest to the feature values to predict with.
                                            If the result is continuous, the result is the mean of nearby neighbours. If the result is categorical, the result is the mode (most common) of nearby neighbours.
                                            "Closeness" of samples is determined by the Euclidean distance on continuous features, plus the Hamming distance on categorical features.
                                            <br></br><br></br>
                                            <i>Linear Regression</i> assumes that the result is a linear combination of the features (and that features are independent of one another).
                                            For continuous features this means determining a line of best fit for one feature, or a plane of best fit for two features, and so on.
                                            For categorical features, the model will determine the best constant to add to the prediction for each possible category.
                                            <br></br><br></br>
                                            <i>Polynomial Regression</i> assumes that the result is some polynomial of a single continuous feature. The model determines the best-fitting polynomial of the specified degree.
                                        </td>
                                    </tr>
                                </table>
                            </p>
                        :(instTab[2]==="active") ?
                            <p>
                                <table border="0">
                                    <tr>
                                        <td align="right" style={{"vertical-align": "top", "white-space": "nowrap"}}>
                                            <b>Features</b>&nbsp;
                                        </td>
                                        <td align="left">
                                            These are the independent values for the model to use in predicting the result; like known 'x' values.
                                            <br></br><br></br>
                                            Select a field as a <i>continuous</i> feature if the field contains numerical data, with a sense of "bigger" or "smaller".
                                            <br></br><br></br>
                                            Select a field as a <i>categorical</i> feature if the field values are options, e.g. colours or classes.
                                            <br></br><br></br>
                                            
                                            Which features you choose are completely up to you! Experiment and see which features give the best accuracy in predicting results.
                                            Be aware of selecting your result as one of your features: this is a form of data leakage. If your model accuracy seems too good to be true, it probably is!
                                            <br></br><br></br>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td align="right" style={{"vertical-align": "top", "white-space": "nowrap"}}>
                                            <b>Result</b>&nbsp;
                                        </td>
                                        <td align="left">
                                            The field of interest that you want the model to predict from your chosen features.
                                        </td>
                                    </tr>
                                </table>
                            </p>
                        :(instTab[3]==="active") ?
                            <p>
                                <table border="0">
                                    <tr>
                                        <td align="right" style={{"vertical-align": "top", "white-space": "nowrap"}}>
                                            <b>Model Map</b>&nbsp;
                                        </td>
                                        <td align="left">
                                            A visual representation of your model. Representations are generated for Decision Trees, KNN with one or two continuous features, Linear Regression models with one or two continuous features, and Polynomial Regression models.
                                            <br></br><br></br>
                                            Representations of Decision Trees are reduced down to a maximum depth of 4 for enhanced visibility, regardless of the actual maximum depth of the model.
                                        </td>
                                    </tr>
                                </table>
                            </p>
                        :(instTab[4]==="active") ?
                            <p>
                                <table border="0">
                                    <tr>
                                        <td align="right" style={{"vertical-align": "top", "white-space": "nowrap"}}>
                                            <b>Model Accuracy</b>&nbsp;
                                        </td>
                                        <td align="left">
                                            A measure of how well your model predicts results, assessed by comparing model predictions with the true results given in the data.
                                            <br></br><br></br>
                                            <i>R2 accuracy</i>, also known as the <i>Coefficient of Determination</i>, measures the accuracy of regression models. The best possible score is 1.0. If the score is 0.0, a constant model which always predicts the mean value of the result values would be just as accurate as your model.
                                            <br></br><br></br>
                                            <i>Classifier accuracy</i> measures the accuracy of classification models. It is the proportion of predictions which are correct. The best possible score is 1.0.
                                            <br></br><br></br>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td align="right" style={{"vertical-align": "top", "white-space": "nowrap"}}>
                                            <b>Data Leakage</b>&nbsp;
                                        </td>
                                        <td align="left">
                                            If your model has a surprisingly high accuracy on both training and testing data, this could be a sign of <i>data leakage</i>, which means your model has used information while predicting that would not normally be available when making predictions in the real world.
                                            <br></br><br></br>
                                            One form of data leakage is <i>Feature Leakage</i>, where information about the result is leaked into the selected features, for example, including an "hoursAwake" feature when trying to predict "hoursAsleep".
                                            <br></br><br></br>
                                            Another form of data leakage is <i>Training Data Leakage</i>, where information is shared between entries in the dataset, meaning your model gets a sneak-peak at the testing data while training. This can happen if there are duplicate entries in the data, or if entries aren't i.i.d.
                                            <br></br><br></br>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td align="right" style={{"vertical-align": "top", "white-space": "nowrap"}}>
                                            <b>Imbalanced Data</b>&nbsp;
                                        </td>
                                        <td align="left">
                                            Surprisingly high accuracies can also be a sign of <i>Imbalanced Data</i>, meaning your data has a minority class in the result. For example, if the result is "Red" 99% of the time, your model might learn to constantly predict "Red" to achieve 99% accuracy.
                                            <br></br><br></br>
                                            While correct almost all of the time, your model will never identify rare results, so accuracy is not the be-all and end-all in assessing models. To improve models on imbalanced data, one can consider downsampling and upweighting larger classes. This is not yet performed in this app.
                                            <br></br><br></br>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td align="right" style={{"vertical-align": "top", "white-space": "nowrap"}}>
                                            <b>Over-Fitting</b>&nbsp;
                                        </td>
                                        <td>
                                            If your model has high accuracy on training data, but low accuracy on testing data, this might indicate <i>over-fitting</i>, whereby your model is overly-complex and hyper-specified to perform well on your training data, but fails to generalise to unseen training data.
                                            <br></br><br></br>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td align="right" style={{"vertical-align": "top", "white-space": "nowrap"}}>
                                            <b>Bias</b>&nbsp;
                                        </td>
                                        <td>
                                            If your model has low accuracy on both training and testing data, your model might be <i>biased</i> or might not be complex enough. Consider providing your model with more (relevant) features.
                                        </td>
                                    </tr>
                                </table>
                            </p>
                        :(instTab[5]==="active") &&
                            <p>
                                <table border="0">
                                    <tr>
                                        <td align="right" style={{"vertical-align": "top", "white-space": "nowrap"}}>
                                            <b>Predicting</b>&nbsp;
                                        </td>
                                        <td align="left">
                                            Enter some feature values for which you would like to predict a result.
                                            <br></br><br></br>
                                            Be aware that if your dataset is small or contains minority classes in some fields, because some data is set aside for testing, there's a chance your model will be trained never having seen some classes, in which case you won't be able to pick them in predictions.
                                            <br></br><br></br>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td align="right" style={{"vertical-align": "top", "white-space": "nowrap"}}>
                                            <b>Prediction</b>&nbsp;
                                        </td>
                                        <td align="left">
                                            The prediction of your model at the specified feature values.
                                            <br></br><br></br>
                                            As for minority classes in features, be aware that if your dataset contains minority classes in the result field, your model might not ever see that result class during training and won't predict that result.
                                        </td>
                                    </tr>
                                </table>
                            </p>
                        }
                </div>
            </div>
        </div>

        <div class="column-container">
            <div class="column">
                <div class="card">
                    <div className="container-header">
                        <h2>Machine Learner Inputs</h2>
                    </div>
                    <div class="container">
                        <h3 align="center">Data Selection</h3>
                        <FormDataset parentCallback = {callbackFunctionDataset} />
                        <br></br>
                        <hr color="#03bffe"></hr>
                        {!(datasetName) && <h3 align="center">Please select a dataset...</h3> }
                        {(datasetName) && <div> <h3 align="center">Model Definition<br></br> {datasetName} </h3> </div> }
                        <FormDefineModel parentCallback = {callbackFunction} isLoadingModelFit = {loadingModelFit} datasetFields = {datasetFields} datasetName = {datasetName}/>
                        <br></br>
                        <hr color="#03bffe"></hr>
                        {!(predictionTitle) && <h3 align="center">Please fit a model...</h3> }
                        {(predictionTitle) && <h3 align="center">Prediction</h3> }
                        <FormPredictAt inputValidation = {inputValidation} datasetResultParam = {datasetResultParam} parentCallback = {callbackFunctionPredict} isLoadingModelPredict = {loadingModelPredict} isLoadingModelFit = {loadingModelFit} predictionTitle={predictionTitle}/>
                        <br></br>
                    </div>
                </div>
            </div>

            <div class="column">
                <div class="card">
                    <div className="container-header">
                        <h2 align="center">Machine Learner Outputs</h2>
                    </div>
                    <div class="container">
                        <h3 align="center">Model Representation</h3>
                        <FormModelOutputs modelOutputs={mlOuts} sessionID={sessionID} />
                        <br></br>
                        <hr color="#03bffe"></hr>
                        <FormModelPrediction modelPrediction={prediction} datasetFeatures = {datasetFeatures} datasetResultParam = {datasetResultParam} />
                        <br></br>
                    </div>
                </div>
            </div>
        </div>

    </body>
    </html>
  );
}

function arrayEquals(a, b) {
    return Array.isArray(a) &&
        Array.isArray(b) &&
        a.length === b.length &&
        a.every((val, index) => val === b[index]);
}