import './pagestyles.css';
import {FormDataset, FormDefineModel, FormModelOutputs, FormModelPrediction} from './forms';
import React, {useState, useEffect, useCallback} from 'react';

export function MLerPage(props) {
    const [instTab, setInstTab] = useState(["active"]);
    const [sessionId, setSessionId] = useState(0);
    const [inputs, setInputs] = useState();
    const [mlOuts, setMlOuts] = useState(0);
    const [predictAt, setPredictAt] = useState({});
    const [prediction, setPrediction] = useState("");
    const [predictionTitle, setPredictionTitle] = useState(false)
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
        setPredictAt({});
        setPrediction("");
    }    
    const callbackFunctionDataset = (formsData) => {
        if ( !(datasetName==formsData[0]) || !(arrayEquals(datasetFields['fields'],formsData[1]['fields'])) ) {
            setDatasetName(formsData[0]);
            setDatasetFields(formsData[1]);
            setDatasetFeatures();
            setMlOuts(0);
            setPredictionTitle(false);
        }
    }
    
    //Set session ID only once, on initial loading
    useEffect(() => {
        setSessionId(Date.now());
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
                    problemtype: inputs[0],
                    mlmethod: inputs[1],
                    polydeg: inputs[2],
                    ctsparams: inputs[3],
                    cateparams: inputs[4],
                    resultparam: inputs[5],
                    testprop: inputs[6],
                    sessionId: sessionId
                })
            }).then(res => res.json())
                .then(data => {
                    setMlOuts({
                        modelMetrics: data.model_metrics,
                        graphImageBase64: data.graph_image_base_64
                    })
                    setInputValidation({
                        features: inputs[3].concat(inputs[4]),
                        noOfCts: inputs[3].length,
                        options: data.allowed_feature_values_for_prediction
                    })
                    setDatasetResultParam(inputs[5])
                    setDatasetFeatures(inputs[3].concat(inputs[4]))
                    setLoadingModelFit(false)
                    setPredictionTitle(true)
            });
        }
    }, [inputs]);

    //Update model prediction and prediction-loading status upon trigger of predictAt changing
    useEffect(() => {
        if (predictionTitle && predictAt && typeof predictAt === 'object' && Object.keys(predictAt).length > 0) {
            setLoadingModelPredict(true);
            fetch('/api/ml/predict', {
                method: 'post',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    predictAt,
                    sessionId: sessionId
                })
            })
                .then(res => res.json())
                .then(data => {
                    setPrediction(data.prediction?.prediction ?? data.prediction)
                    setLoadingModelPredict(false)
                })
        }
    }, [predictAt])

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
                sessionId: sessionId
            })
        })
        return ''; // Legacy method for cross browser support
      };

    //ML page, i.e. Parent, receives form inputs from Model Definition form, i.e. Child 1
    //Then hook is triggered by the state change in the inputs variable, which executes the python function via the api
    //Which updates the results, e.g. accuracy of the model, image...
    //These results are then passes from Parent to Output Forms, i.e. Child 2, for display

    // Helper for section tooltips
    const TooltipIcon = ({ text }) => (
        <span className="tooltip-wrapper">
            <span className="tooltip-icon">ℹ</span>
            <span className="tooltip-text">{text}</span>
        </span>
    );

    // Ghost placeholder component for unavailable content
    const GhostPlaceholder = ({ icon, title, description }) => (
        <div className="ghost-placeholder">
            <div className="ghost-icon">{icon}</div>
            <h4 className="ghost-title">{title}</h4>
            <p className="ghost-description">{description}</p>
        </div>
    );

    return (
    <div className="ml-page">

    <div className="about-section">
            <h1>Machine Learner</h1>
        </div>

        {/* Instructions Card - kept as-is */}
        <div className="instructions-card">
            <div className="container-header-bar">
                <a className={instTab[0]} onClick={() => setInstTab(["active"]) }>Getting Started</a>
                <a className={instTab[1]} onClick={() => setInstTab( Array(2).fill("").fill("active",-1) ) }>Data Selection</a>
                <a className={instTab[2]} onClick={() => setInstTab( Array(3).fill("").fill("active",-1) ) }>Model Definition</a>
                <a className={instTab[3]} onClick={() => setInstTab( Array(4).fill("").fill("active",-1) ) }>Features & Result</a>
                <a className={instTab[4]} onClick={() => setInstTab( Array(5).fill("").fill("active",-1) ) }>Model Representation</a>
                <a className={instTab[5]} onClick={() => setInstTab( Array(6).fill("").fill("active",-1) ) }>Accuracy</a>
                <a className={instTab[6]} onClick={() => setInstTab( Array(7).fill("").fill("active",-1) ) }>Prediction</a>
            </div>
            <div className="container">
                { (instTab[0]==="active") ?
                        <div>
                            <p className="info-text">
                                This app is a general machine learner: designed to be intuitive, and built to allow you to fit a machine-led model to any data you like!
                            </p>
                            <p className="info-text">
                                The stages involved in using this app are: choosing a dataset, configuring model parameters, selecting features and the result to predict, reviewing the model representation and accuracy, and using your model to predict results.
                            </p>
                            <p className="info-text">
                                Click through these tabs for instructions to help you through each stage of building your models. Alternatively, get started below using the tooltip icons for help!
                            </p>
                        </div>
                    : (instTab[1]==="active") ?
                        <div>
                            <div className="info-section">
                                <h4>Upload Your Data</h4>
                                <p>Upload any CSV dataset where rows are entries, columns are fields, and column headers are included.</p>
                            </div>
                            <div className="info-section">
                                <h4>Default Dataset</h4>
                                <p>Select the default dataset, which contains vehicle emissions data such as make, model, engine size, and CO2 emissions (g/km).</p>
                                <p>The raw data was obtained from the UK government's vehicle certification agency at <a href="https://www.gov.uk/governmentorganisations/vehicle-certification-agency" target="_blank" rel="noopener noreferrer">https://www.gov.uk/governmentorganisations/vehicle-certification-agency</a>. I then cleaned this data as part of my ML investigation into CO2 emissions at <a href="https://github.com/ReubenGitHub/ML-Vehicle-Emissions" target="_blank" rel="noopener noreferrer">https://github.com/ReubenGitHub/ML-Vehicle-Emissions</a>.</p>
                            </div>
                        </div>
                    : (instTab[2]==="active") ?
                        <div>
                            <div className="info-section">
                                <h4>Problem Type</h4>
                                <p>Select <i>Regression</i> if the result you want to predict is a continuous variable, i.e. numerical and where some values can be "bigger" than others.</p>
                                <p>Select <i>Classification</i> if the result you want to predict is a categorical variable, i.e. categories or types of something, with no sense of "bigger" or "smaller".</p>
                            </div>
                            <div className="info-section">
                                <h4>Machine Learning Method</h4>
                                <p><i>Decision Trees</i> will derive a prediction through a sequence of decisions based on known features, similar to the game '20Q'.</p>
                                <p><i>K-Nearest Neighbours</i> predicts a result by averaging the results of K training samples which have feature values closest to the feature values to predict with.
                                If the result is continuous, the result is the mean of nearby neighbours. If the result is categorical, the result is the mode (most common) of nearby neighbours.
                                "Closeness" of samples is determined by the Euclidean distance on continuous features, plus the Hamming distance on categorical features.</p>
                                <p><i>Linear Regression</i> assumes that the result is a linear combination of the features (and that features are independent of one another).
                                For continuous features this means determining a line of best fit for one feature, or a plane of best fit for two features, and so on.
                                For categorical features, the model will determine the best constant to add to the prediction for each possible category.</p>
                                <p><i>Polynomial Regression</i> assumes that the result is some polynomial of a single continuous feature. The model determines the best-fitting polynomial of the specified degree.</p>
                            </div>
                        </div>
                    : (instTab[3]==="active") ?
                        <div>
                            <div className="info-section">
                                <h4>Features</h4>
                                <p>These are the independent values for the model to use in predicting the result; like known 'x' values.</p>
                                <p>Select a field as a <i>continuous</i> feature if the field contains numerical data, with a sense of "bigger" or "smaller".</p>
                                <p>Select a field as a <i>categorical</i> feature if the field values are options, e.g. colours or classes.</p>
                                <p>Which features you choose are completely up to you! Experiment and see which features give the best accuracy in predicting results.
                                Be aware of selecting your result as one of your features: this is a form of data leakage. If your model accuracy seems too good to be true, it probably is!</p>
                            </div>
                            <div className="info-section">
                                <h4>Result</h4>
                                <p>The field of interest that you want the model to predict from your chosen features.</p>
                            </div>
                        </div>
                    : (instTab[4]==="active") ?
                        <div>
                            <div className="info-section">
                                <h4>Model Map</h4>
                                <p>A visual representation of your model. Representations are generated for Decision Trees, KNN with one or two continuous features, Linear Regression models with one or two continuous features, and Polynomial Regression models.</p>
                                <p>Representations of Decision Trees are reduced down to a maximum depth of 4 for enhanced visibility, regardless of the actual maximum depth of the model.</p>
                            </div>
                        </div>
                    : (instTab[5]==="active") ?
                        <div>
                            <div className="info-section">
                                <h4>Model Accuracy</h4>
                                <p>A measure of how well your model predicts results, assessed by comparing model predictions with the true results given in the data.</p>
                                <p><i>R2 accuracy</i>, also known as the <i>Coefficient of Determination</i>, measures the accuracy of regression models. The best possible score is 1.0. If the score is 0.0, a constant model which always predicts the mean value of the result values would be just as accurate as your model.</p>
                                <p><i>Classifier accuracy</i> measures the accuracy of classification models. It is the proportion of predictions which are correct. The best possible score is 1.0.</p>
                            </div>
                            <div className="info-section">
                                <h4>Data Leakage</h4>
                                <p>If your model has a surprisingly high accuracy on both training and testing data, this could be a sign of <i>data leakage</i>, which means your model has used information while predicting that would not normally be available when making predictions in the real world.</p>
                                <p>One form of data leakage is <i>Feature Leakage</i>, where information about the result is leaked into the selected features, for example, including an "hoursAwake" feature when trying to predict "hoursAsleep".</p>
                                <p>Another form of data leakage is <i>Training Data Leakage</i>, where information is shared between entries in the dataset, meaning your model gets a sneak-peak at the testing data while training. This can happen if there are duplicate entries in the data, or if entries aren't i.i.d.</p>
                            </div>
                            <div className="info-section">
                                <h4>Imbalanced Data</h4>
                                <p>Surprisingly high accuracies can also be a sign of <i>Imbalanced Data</i>, meaning your data has a minority class in the result. For example, if the result is "Red" 99% of the time, your model might learn to constantly predict "Red" to achieve 99% accuracy.</p>
                                <p>While correct almost all of the time, your model will never identify rare results, so accuracy is not the be-all and end-all in assessing models. To improve models on imbalanced data, one can consider downsampling and upweighting larger classes. This is not yet performed in this app.</p>
                            </div>
                            <div className="info-section">
                                <h4>Over-Fitting</h4>
                                <p>If your model has high accuracy on training data, but low accuracy on testing data, this might indicate <i>over-fitting</i>, whereby your model is overly-complex and hyper-specified to perform well on your training data, but fails to generalise to unseen training data.</p>
                            </div>
                            <div className="info-section">
                                <h4>Bias</h4>
                                <p>If your model has low accuracy on both training and testing data, your model might be <i>biased</i> or might not be complex enough. Consider providing your model with more (relevant) features.</p>
                            </div>
                        </div>
                    : (instTab[6]==="active") &&
                        <div>
                            <div className="info-section">
                                <h4>Predicting</h4>
                                <p>Enter some feature values for which you would like to predict a result.</p>
                                <p>Be aware that if your dataset is small or contains minority classes in some fields, because some data is set aside for testing, there's a chance your model will be trained never having seen some classes, in which case you won't be able to pick them in predictions.</p>
                            </div>
                            <div className="info-section">
                                <h4>Prediction</h4>
                                <p>The prediction of your model at the specified feature values.</p>
                                <p>As for minority classes in features, be aware that if your dataset contains minority classes in the result field, your model might not ever see that result class during training and won't predict that result.</p>
                            </div>
                        </div>
                    }
            </div>
        </div>

        {/* Sequential Workflow Sections */}
        <div className="ml-workflow">
            
            {/* Section 1: Data Selection */}
            <div className="ml-section">
                <div className="ml-section-header">
                    <h2>1. Data Selection <TooltipIcon text="Choose a dataset to train your model on. The default dataset contains vehicle CO2 emissions data." /></h2>
                </div>
                <div className="ml-section-content">
                    <FormDataset parentCallback={callbackFunctionDataset} sessionId={sessionId} />
                </div>
            </div>

            {/* Section 2: Model Definition */}
            <div className="ml-section">
                <div className="ml-section-header">
                    <h2>2. Model Definition <TooltipIcon text="Configure your model parameters. Choose supervision type, problem type, and ML method." /></h2>
                </div>
                <div className="ml-section-content">
                    {!datasetName ? (
                        <GhostPlaceholder 
                            icon="⚙️" 
                            title="Select a Dataset First" 
                            description="Choose a dataset in step 1 before defining your model."
                        />
                    ) : (
                        <>
                            <FormDefineModel 
                                key="defineModel" 
                                parentCallback={callbackFunction} 
                                isLoadingModelFit={loadingModelFit} 
                                datasetFields={datasetFields} 
                                datasetName={datasetName}
                                tooltips={{
                                    problemType: "Select Regression for continuous numerical results (e.g. price, temperature). Select Classification for categorical results (e.g. yes/no, red/blue/green).",
                                    modelType: "Decision Trees: sequence of decisions like '20Q'. KNN: averages nearest training samples. Linear Regression: best-fit line/plane. Polynomial Regression: best-fitting curve of specified degree.",
                                    continuousFeatures: "Select fields with numerical data where 'bigger' or 'smaller' makes sense (e.g. age, height, weight).",
                                    categoricalFeatures: "Select fields with categories or options (e.g. color, type, brand). Not available for Polynomial Regression.",
                                    result: "The field you want the model to predict. Warning: don't select a feature that contains information about the result (data leakage).",
                                    testProportion: "Percentage of data set aside for testing model accuracy. Common values are 20-30%. More test data gives more reliable accuracy estimates but less data for training."
                                }}
                            />
                            
                            {/* Model Representation & Metrics - shown after model is fitted */}
                            {predictionTitle && mlOuts?.modelMetrics ? (
                                <div className="ml-results-panel">
                                    <h3>Model Representation & Metrics</h3>
                                    <FormModelOutputs modelOutputs={mlOuts} sessionId={sessionId} />
                                </div>
                            ) : (
                                <GhostPlaceholder 
                                    icon="📈" 
                                    title="Fit a Model to See Results" 
                                    description="Configure and fit your model to see the representation and accuracy metrics."
                                />
                            )}
                        </>
                    )}
                </div>
            </div>

            {/* Section 3: Prediction */}
            <div className="ml-section">
                <div className="ml-section-header">
                    <h2>3. Prediction <TooltipIcon text="Use your trained model to predict results for new feature values." /></h2>
                </div>
                <div className="ml-section-content">
                    {!predictionTitle ? (
                        <GhostPlaceholder 
                            icon="🔮" 
                            title="Train Your Model First" 
                            description="Complete steps 1 and 2, then fit your model before making predictions."
                        />
                    ) : (
                        <FormModelPrediction
                            key="modelPrediction"
                            modelPrediction={{ predictAt: predictAt, prediction: prediction }}
                            datasetFeatures={datasetFeatures}
                            datasetResultParam={datasetResultParam}
                            inputValidation={inputValidation}
                            onPredict={(values) => setPredictAt(values)}
                        />
                    )}
                </div>
            </div>

        </div>

    </div>
  );
}

function arrayEquals(a, b) {
    return Array.isArray(a) &&
        Array.isArray(b) &&
        a.length === b.length &&
        a.every((val, index) => val === b[index]);
}
