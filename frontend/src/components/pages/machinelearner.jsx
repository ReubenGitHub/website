import '../global.css';
import '../headernavbar.css';
import '../Hero.css';
import '../home.css';
import '../forms/forms.css';
import '../machinelearner.css';
import {FormDataset} from '../forms/FormDataset';
import {FormDefineModel} from '../forms/FormDefineModel';
import {FormModelOutputs} from '../forms/FormModelOutputs';
import {FormModelPrediction} from '../forms/FormModelPrediction';
import {InstructionsGettingStarted} from '../instructions/InstructionsGettingStarted';
import {InstructionsDataSelection} from '../instructions/InstructionsDataSelection';
import {InstructionsModelDefinition} from '../instructions/InstructionsModelDefinition';
import {InstructionsFeaturesResult} from '../instructions/InstructionsFeaturesResult';
import {InstructionsModelRepresentation} from '../instructions/InstructionsModelRepresentation';
import {InstructionsAccuracy} from '../instructions/InstructionsAccuracy';
import {InstructionsPrediction} from '../instructions/InstructionsPrediction';
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
                { instTab[0]==="active" && <InstructionsGettingStarted /> }
                { instTab[1]==="active" && <InstructionsDataSelection /> }
                { instTab[2]==="active" && <InstructionsModelDefinition /> }
                { instTab[3]==="active" && <InstructionsFeaturesResult /> }
                { instTab[4]==="active" && <InstructionsModelRepresentation /> }
                { instTab[5]==="active" && <InstructionsAccuracy /> }
                { instTab[6]==="active" && <InstructionsPrediction /> }
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
