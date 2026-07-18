import React, {useState, useEffect} from 'react';
import './forms.css';
import representationIcon from '../images/repres_icon.png';
import representationNAIcon from '../images/repres_na_icon.png';

const DEFAULT_FILENAME = "Default: CO2 Emissions.csv"

export function FormDataset(props) {
    
    const [dataset, setDataset] = useState()
    const [datasetName, setDatasetName] = useState()
    const [datasetFields, setDatasetFields] = useState({fields: '', nonCtsFields: ''})
    const [datasetIsUpload, setDatasetIsUpload] = useState(false)
    const [count, setCount] = useState(0)
    

    const handleSubmit = () => {
        const fetchDataset = (useDefaultDataset, dataset) => {
            fetch('/api/uploadDataset', {
                method: 'post',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    useDefaultDataset,
                    ...(useDefaultDataset ? {} : { dataset }),
                    sessionId: props.sessionId
                })
            })
                .then(res => res.json())
                .then(data => {
                    if (data.datasetFields) {
                        setDatasetFields(data.datasetFields)
                    } else {
                        alert('Invalid dataset')
                    }
                })
        }
        
        if (!datasetIsUpload) {
            setDatasetName(DEFAULT_FILENAME)
            fetchDataset(true)
        } else {
            const file = dataset
            const reader = new FileReader()

            reader.onload = (e) => {
                const text = e.target.result
                const filename = file["name"]
                const maxFileSize = 2000000
            
                if (file["size"] > maxFileSize) {
                    alert('Please upload a dataset no bigger than ' + maxFileSize/1000000 + 'MB  :)')
                } else {
                    setDatasetName(filename)
                    fetchDataset(false, text)
                }
            }
        
            reader.readAsText(file)
        }
    }

    useEffect(() => { 
        if (count) {
            props.parentCallback( [datasetName, datasetFields] )
        }
        setCount(count+1);
    }, [datasetFields]);

    return (
        <form onSubmit={ (e) => {
                e.preventDefault()
                if(dataset || datasetIsUpload==false)handleSubmit()
            }}>
            <b>Select a .csv Dataset </b>
            <div className="dataset-option-group">
                <input type="radio" id="datasetDefault" name="datasetSelect" required defaultChecked onChange={(e) => {setDatasetIsUpload(false); setDataset()} } />
                <label className="dataset-option" htmlFor="datasetDefault">
                    <span className="dataset-option-icon">🚗</span>
                    <span className="dataset-option-text">
                        <span className="dataset-option-title">Default Dataset</span>
                        <span className="dataset-option-desc">Vehicle CO2 emissions data</span>
                    </span>
                </label>

                <input type="radio" id="datasetUpload" name="datasetSelect" required onChange={(e) => setDatasetIsUpload(true)} />
                <label className="dataset-option" htmlFor="datasetUpload">
                    <span className="dataset-option-icon">📁</span>
                    <span className="dataset-option-text">
                        <span className="dataset-option-title">Upload Your Own</span>
                        <span className="dataset-option-desc">CSV file, max 2MB</span>
                    </span>
                </label>
            </div>

            {datasetIsUpload && (
                <div className="glass-file-input">
                    <input type="file" accept=".csv" id="dataset" onChange={(e) => setDataset(e.target.files[0])} />
                    <label htmlFor="dataset" className="glass-file-label">
                        <span className="glass-file-icon">📄</span>
                        <span className="glass-file-text">Choose a CSV file</span>
                    </label>
                </div>
            )}

            <br></br>
            <div className="ml-button-container">
                { (!datasetIsUpload || dataset) ?
                    <button className="ml-button">Commit Dataset</button> :
                    <button disabled className="ml-button">Commit Dataset</button> }
            </div>

            {datasetName && (
                <div className="dataset-badge">
                    <span className="dataset-badge-icon">📄</span>
                    <span>Dataset selected: "{datasetName}"</span>
                </div>
            )}
        </form>
    )
}

export function FormDefineModel(props) {
    const [MLMethod, setMLMethod] = useState("DT");
    const [PolyDeg, setPolyDeg] = useState(2);
    const [ctsParams, setCtsParams] = useState("");
    const [cateParams, setCateParams] = useState("");
    const [resultParam, setResultParam] = useState("");
    const [problemType, setProblemType] = useState("");
    const [testProp, setTestProp] = useState(20);
    const [inputs, setInputs] = useState();
    const [datasetFields, setDatasetFields] = useState({fields: ["Fields..."], nonCtsFields: []});
    const [datasetFieldsNo, setDatasetFieldsNo] = useState(["0"]);
    const [count, setCount] = useState(0);
    
    const handleSubmit = (event) => {
        event.preventDefault();
        var ctsFeatures = [];
        var cateFeatures = [];
        for(let i = 0; i < datasetFields['fields'].length; i++) {
            if (ctsParams[i]) {
                ctsFeatures.push(datasetFields['fields'][i]);
            }
            if (cateParams[i]) {
                cateFeatures.push(datasetFields['fields'][i]);
            }
        }
        if (ctsFeatures.every(v => v===false) && cateFeatures.every(v => v===false)) {
            alert('Please select at least one feature in the Machine Learner Inputs  :)');
        } else {
            setInputs([problemType, MLMethod, Number(PolyDeg), ctsFeatures, cateFeatures, resultParam, testProp/100])
        }
    }

    useEffect(() => {
        if (props.datasetFields['fields']) {
            if (!arrayEquals(datasetFields['fields'],props.datasetFields['fields'])) {
                setDatasetFields(props.datasetFields);
                //setDatasetFieldsNo( [...Array( props.datasetFields.length).keys()].filter(x => x % 2 == 0)); //For checkboxes table setup
                setDatasetFieldsNo( [...Array( props.datasetFields['fields'].length).keys()] );
                setCtsParams( Array(props.datasetFields['fields'].length).fill(false) );
                setCateParams( Array(props.datasetFields['fields'].length).fill(false) );
                setResultParam("");
                document.querySelectorAll('input[att=clearOnDataCommit]').forEach( el => el.checked = false );
            }
        }
    }, [props.datasetFields]);

    useEffect(() => { //useEffect because setInputs is asynchronous. this ensures only pass inputs up to ML page once ready
        if (count) {
            props.parentCallback(inputs)
        }
        setCount(count+1);
    }, [inputs]);

    return (
        <form onSubmit={handleSubmit}>
            <b>Problem Type</b>
            <div className="pill-selector">
                <input type="radio" id="regression" name="problemType" value="regression" required onChange={(e) => {
                    setProblemType(e.target.value);
                    { ( props.datasetFields['nonCtsFields'].includes(resultParam) ) && setResultParam("") }
                    { ( props.datasetFields['nonCtsFields'].includes(resultParam) ) && document.querySelectorAll('input[att2=clearOnRegression]').forEach( el => el.checked = false ) }
                }} />
                <label className={problemType === "regression" ? "pill-btn" : "pill-btn"} htmlFor="regression">Regression</label>

                <input type="radio" id="classification" name="problemType" value="classification" required onChange={(e) => {
                    setProblemType(e.target.value);
                    if (MLMethod !== "DT" && MLMethod !== "KNN") {
                        setMLMethod("DT");
                    }
                }} />
                <label className={problemType === "classification" ? "pill-btn" : "pill-btn"} htmlFor="classification">Classification</label>
            </div>
            <br></br>
            <b>Model Type</b>
            <br></br>
            <div className="model-type-wrapper">
                { (problemType==="") ?
                        <select
                        disabled
                        value={MLMethod}
                        onChange={(e) => setMLMethod(e.target.value)}
                    >
                        <option value="DT">Decision Tree</option>
                    </select>
                    : (problemType==="regression") ?
                        <select
                            value={MLMethod}
                            onChange={(e) => {
                                setMLMethod(e.target.value);
                                {(e.target.value==="PolyFit") && setCateParams( Array(props.datasetFields['fields'].length).fill(false) ) }
                                {(e.target.value==="PolyFit") && setCtsParams( Array(props.datasetFields['fields'].length).fill(false) ) }
                                {(e.target.value==="PolyFit") && document.querySelectorAll('input[att3=clearOnPolyFit]').forEach( el => el.checked = false ) }
                            }}
                        >
                            <option value="DT">Decision Tree</option>
                            <option value="KNN">K-Nearest Neighbours</option>
                            <option value="LinReg">Linear Regression</option>
                            <option value="PolyFit">Polynomial Regression</option>
                        </select>
                    :(problemType==="classification") &&
                        <select
                            value={MLMethod}
                            onChange={(e) => setMLMethod(e.target.value)}
                        >
                            <option value="DT">Decision Tree</option>
                            <option value="KNN">K-Nearest Neighbours</option>
                            <option value="LinReg" disabled style={{color: "#989897"}}>Linear Regression</option>
                            <option value="PolyFit" disabled style={{color: "#989897"}}>Polynomial Regression</option>
                        </select>
                }
            </div>
            <br></br>
            { (MLMethod=="PolyFit") && 
                <label> Degrees &nbsp;
                    <input
                        type="number"
                        min="0"
                        max="99"
                        value={PolyDeg}
                        required
                        onChange={(e) => setPolyDeg(e.target.value)}
                    />
                    <br></br>
                </label>
            }
            <br></br>
            <b>Features - Continuous</b>
            <div className="pill-selector">
                { datasetFieldsNo.map( (index) => {
                    const fieldName = datasetFields['fields'][index];
                    const isDisabled = !(props.datasetFields['fields']) || props.datasetFields['nonCtsFields'].includes(fieldName);
                    const isPolyFit = MLMethod === "PolyFit";
                    const inputType = isPolyFit ? "radio" : "checkbox";
                    const pillClass = isDisabled ? "pill-btn pill-disabled" : "pill-btn";
                    const inputProps = {
                        id: fieldName + "cts",
                        type: inputType,
                        name: "continuous features",
                        att: "clearOnDataCommit",
                        att3: "clearOnPolyFit",
                        className: "pill-input"
                    };
                    
                    if (isDisabled) {
                        return (
                            <React.Fragment key={'cts-'+index}>
                                <input {...inputProps} disabled />
                                <label className="pill-btn pill-disabled" htmlFor={fieldName + "cts"}>
                                    {fieldName.length > 15 ? fieldName.substring(0, 15) + '...' : fieldName}
                                </label>
                            </React.Fragment>
                        );
                    }
                    
                    return (
                        <React.Fragment key={'cts-'+index}>
                            <input 
                                {...inputProps}
                                onChange={(e) => {
                                    const newCtsParams = [...ctsParams];
                                    const newCateParams = [...cateParams];
                                    if (isPolyFit) {
                                        for (let i=0; i<newCtsParams.length; ++i) {
                                            newCtsParams[i] = false;
                                        }
                                        newCtsParams[index] = e.target.checked;
                                    } else {
                                        newCtsParams[index] = e.target.checked;
                                        newCateParams[index] = false;
                                        document.querySelectorAll('input[id="'+fieldName+'ctg"]').forEach( el => el.checked = false );
                                    }
                                    setCtsParams(newCtsParams);
                                    setCateParams(newCateParams);
                                }}
                            />
                            <label className={pillClass} htmlFor={fieldName + "cts"}>
                                {fieldName.length > 15 ? fieldName.substring(0, 15) + '...' : fieldName}
                            </label>
                        </React.Fragment>
                    );
                }) }
            </div>
            <br></br>
            <b>Features - Categorical</b>
            <div className="pill-selector">
                { datasetFieldsNo.map( (index) => {
                    const fieldName = datasetFields['fields'][index];
                    const isDisabled = !(props.datasetFields['fields']) || MLMethod === "PolyFit";
                    const pillClass = isDisabled ? "pill-btn pill-disabled" : "pill-btn";
                    
                    if (isDisabled) {
                        return (
                            <React.Fragment key={'ctg-'+index}>
                                <input type="checkbox" disabled id={fieldName + "ctg"} name="categorical features" att="clearOnDataCommit" att3="clearOnPolyFit" className="pill-input" />
                                <label className="pill-btn pill-disabled" htmlFor={fieldName + "ctg"}>
                                    {fieldName.length > 15 ? fieldName.substring(0, 15) + '...' : fieldName}
                                </label>
                            </React.Fragment>
                        );
                    }
                    
                    return (
                        <React.Fragment key={'ctg-'+index}>
                            <input 
                                type="checkbox"
                                id={fieldName + "ctg"}
                                name="categorical features"
                                att="clearOnDataCommit"
                                att3="clearOnPolyFit"
                                className="pill-input"
                                onChange={(e) => {
                                    cateParams[index] = e.target.checked;
                                    ctsParams[index] = false;
                                    document.querySelectorAll('input[id="'+fieldName+'cts"]').forEach( el => el.checked = false );
                                }}
                            />
                            <label className={pillClass} htmlFor={fieldName + "ctg"}>
                                {fieldName.length > 15 ? fieldName.substring(0, 15) + '...' : fieldName}
                            </label>
                        </React.Fragment>
                    );
                }) }
            </div>
            <br></br>
            <b>Result</b>
            <div className="pill-selector">
                { datasetFieldsNo.map( (index) => {
                    const fieldName = datasetFields['fields'][index];
                    const isDisabled = !(props.datasetFields['fields']) || (props.datasetFields['nonCtsFields'].includes(fieldName) && problemType === "regression");
                    const pillClass = isDisabled ? "pill-btn pill-disabled" : "pill-btn";
                    const isSelected = resultParam === fieldName;
                    
                    if (isDisabled) {
                        return (
                            <React.Fragment key={'result-'+index}>
                                <input type="radio" disabled id={fieldName} name="result" att="clearOnDataCommit" att2="clearOnRegression" className="pill-input" />
                                <label className="pill-btn pill-disabled" htmlFor={fieldName}>
                                    {fieldName.length > 15 ? fieldName.substring(0, 15) + '...' : fieldName}
                                </label>
                            </React.Fragment>
                        );
                    }
                    
                    return (
                        <React.Fragment key={'result-'+index}>
                            <input 
                                type="radio"
                                id={fieldName}
                                name="result"
                                att="clearOnDataCommit"
                                att2="clearOnRegression"
                                required
                                className="pill-input"
                                checked={isSelected}
                                onChange={(e) => setResultParam(e.target.id)}
                            />
                            <label className={isSelected ? "pill-btn" : pillClass} htmlFor={fieldName}>
                                {fieldName.length > 15 ? fieldName.substring(0, 15) + '...' : fieldName}
                            </label>
                        </React.Fragment>
                    );
                }) }
            </div>
            <br></br>
            <b>Test Data Proportion</b>
            <br></br>
            <label>
                <input
                    type="number"
                    min="1"
                    max="99"
                    value={testProp}
                    required
                    onChange={(e) => setTestProp(e.target.value)}
                    className="test-proportion-input"
                />
                <span className="test-proportion-suffix">%</span>
            </label>
            <br></br>
            <div className="ml-button-container">
                { props.isLoadingModelFit ? <button disabled className="ml-button">Fitting Model...</button>:
                    !(props.datasetName) ? <button disabled className="ml-button">Fit Model</button>:
                    <button className="ml-button">Fit Model</button>
                }
            </div>  
        </form>
    )
    
}

export function FormPredictAt(props) {
    const [datasetFeatures, setDatasetFeatures] = useState(["Features..."]);
    const [datasetFeaturesNo, setDatasetFeaturesNo] = useState(["0"])
    const [predictAt, setPredictAt] = useState([""]);
    const [predictValues, setPredictValues] = useState({});
    const [predictEnable, setPredictEnable] = useState(0);
    const [inputValidation, setInputValidation] = useState({features: ["Features..."], noOfFeatures: ["0"], noOfCts: 0, options: [["Option"]]});

    const handleSubmit = (event) => {
        event.preventDefault();
        // if (predictValues.includes("")) {
        //     const emptyIndex = getIndex(predictValues, "")
        //     alert('Please provide a value for: ' + datasetFeatures[emptyIndex] );
        // } else {
        setPredictAt( predictValues);
        setPredictEnable( predictEnable+1 );
        // }
    }

    useEffect(() => {
        if (props.inputValidation['features']) {
            if (!arrayEquals(inputValidation,props.inputValidation)) {
                setInputValidation({
                    features: props.inputValidation['features'],
                    noOfFeatures: [...Array( props.inputValidation['features'].length).keys()],   //For checkboxes table setup
                    noOfCts: props.inputValidation['noOfCts'],
                    options: props.inputValidation['options']
                });
                setPredictValues(props.inputValidation['features'].reduce(
                    (acc, feature) => ({ ...acc, [feature]: "" }),
                    {}
                ))
                document.querySelectorAll('input[att=clearOnFeatureSelect],select[att=clearOnFeatureSelect]').forEach( el => el.value = "" );
            }
        } else {
            setInputValidation({
                features: ["Features..."],
                noOfFeatures: [...Array( 1).keys()],   //For checkboxes table setup
                noOfCts: 0,
                options: [["Option"]]
            });
            setPredictValues({});
            document.querySelectorAll('input[att=clearOnFeatureSelect],select[att=clearOnFeatureSelect]').forEach( el => el.value = "" );
        }
    }, [props.inputValidation]);

    useEffect(() => { //useEffect because setInputs is asynchronous. this ensures only pass inputs up to ML page once ready
        if (predictEnable>0) {
            props.parentCallback(predictAt)
        }
        setPredictEnable( 0 );
    }, [predictEnable]);

    // ---------------------------------------------------------------------------PROBABLY need to use a loop to define initial values for categorical params, or make it display blank
    return (
        <form onSubmit={handleSubmit}>
            <b>Predict a result at</b>
            <br></br>
            <table border="0">
                <tbody>
                { (inputValidation['noOfFeatures']).map( (index) => (
                    <tr key={index}>
                        <td align="right">
                            <label htmlFor={inputValidation['features'][index]}> {inputValidation['features'][index]} </label>
                        </td>
                        <td align="left">
                                {(!props.inputValidation['features']) ? <input id={inputValidation['features'][index]} type="text" disabled ></input> 
                                    // :(index<props.inputValidation['noOfCts']) ?
                                    :(index<inputValidation['noOfCts']) ?
                                        <input
                                        id={inputValidation['features'][index]}
                                        type="number"
                                        step="any"
                                        att="clearOnFeatureSelect"
                                        required
                                        onChange={(e) => predictValues[e.target.id] = e.target.value }
                                        />
                                    :<select id={inputValidation['features'][index]}
                                        att="clearOnFeatureSelect"
                                        style={{width: "100%"}}
                                        required
                                        onChange={(e) => predictValues[e.target.id] = e.target.value} >
                                        <option value="" selected></option>
                                        { (inputValidation['options'][(index-inputValidation['noOfCts'])]).map( (option) => (
                                            <option value={option}>{option}</option>
                                        )) }
                                        </select>
                                }
                        </td>
                    </tr>
                )) }
                </tbody>
            </table>

            <br></br>
            { props.isLoadingModelPredict ? <button disabled className="ml-button ml-button-predict">Predicting {props.datasetResultParam}...</button>:
                props.isLoadingModelFit ? <button disabled className="ml-button ml-button-predict">Predict</button>:
                !(props.predictionTitle) ? <button disabled className="ml-button ml-button-predict">Predict</button>:
                <button className="ml-button ml-button-predict">Predict {props.datasetResultParam}</button>
            } 
        </form>
    )
}

export function FormModelOutputs(props) {
    const { modelMetrics, graphImageBase64 } = props.modelOutputs

    const repImageSrc = !modelMetrics?.train_accuracy
        ? representationIcon
        : !graphImageBase64
            ? representationNAIcon
            : `data:image/png;base64,${graphImageBase64}`

    const isClassification = !!modelMetrics?.train_macro_precision;
    const accuracyLabel = isClassification ? 'Classifier Accuracy' : 'R-Squared Accuracy';

    return (
        <div className="ml-results-container">
            <div className="ml-results-image">
                <img src={repImageSrc} alt="Model Representation" />
            </div>
            <div className="ml-results-metrics">
                <div className="metric-cards">
                    {/* Accuracy Card */}
                    <div className="metric-card">
                        <div className="metric-card-header">{accuracyLabel}</div>
                        <div className="metric-card-row">
                            <span className="metric-card-label">Training</span>
                            <span className="metric-card-value">
                                {!(modelMetrics?.train_accuracy == null) ?
                                    (1 * modelMetrics?.train_accuracy + Number.EPSILON).toFixed(3) :
                                    "..."}
                            </span>
                        </div>
                        <div className="metric-card-divider"></div>
                        <div className="metric-card-row">
                            <span className="metric-card-label">Testing</span>
                            <span className="metric-card-value">
                                {!(modelMetrics?.test_accuracy == null) ?
                                    (1 * modelMetrics?.test_accuracy + Number.EPSILON).toFixed(3) :
                                    "..."}
                            </span>
                        </div>
                    </div>

                    {/* Precision Card (Classification only) */}
                    {isClassification && (
                        <div className="metric-card">
                            <div className="metric-card-header">Precision</div>
                            <div className="metric-card-row">
                                <span className="metric-card-label">Train Macro</span>
                                <span className="metric-card-value">
                                    {(1 * modelMetrics?.train_macro_precision + Number.EPSILON).toFixed(3)}
                                </span>
                            </div>
                            <div className="metric-card-divider"></div>
                            <div className="metric-card-row">
                                <span className="metric-card-label">Train Micro</span>
                                <span className="metric-card-value">
                                    {(1 * modelMetrics?.train_micro_precision + Number.EPSILON).toFixed(3)}
                                </span>
                            </div>
                        </div>
                    )}

                    {/* Recall Card (Classification only) */}
                    {isClassification && (
                        <div className="metric-card">
                            <div className="metric-card-header">Recall</div>
                            <div className="metric-card-row">
                                <span className="metric-card-label">Train Macro</span>
                                <span className="metric-card-value">
                                    {(1 * modelMetrics?.train_macro_recall + Number.EPSILON).toFixed(3)}
                                </span>
                            </div>
                            <div className="metric-card-divider"></div>
                            <div className="metric-card-row">
                                <span className="metric-card-label">Train Micro</span>
                                <span className="metric-card-value">
                                    {(1 * modelMetrics?.train_micro_recall + Number.EPSILON).toFixed(3)}
                                </span>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )

}

//NEEDS TO WIPE PREDICTION UPON DATASET/MODEL/INPUTS CHANGE
export function FormModelPrediction(props) {
    const [datasetFeatures, setDatasetFeatures] = useState([]);
    const [datasetResultParam, setDatasetResultParam] = useState("");
    const [datasetFeaturesNo, setDatasetFeaturesNo] = useState([]);
    const [predictAt, setPredictAt] = useState({});
    const [prediction, setPrediction] = useState("");
    const [inputValues, setInputValues] = useState({});
    const [noOfCts, setNoOfCts] = useState(0);
    const [options, setOptions] = useState([[]]);

    useEffect(() => {
        if (props.modelPrediction && props.modelPrediction['predictAt'] && typeof props.modelPrediction['predictAt'] === 'object' && !Array.isArray(props.modelPrediction['predictAt']) && Object.keys(props.modelPrediction['predictAt']).length > 0) {
            const newPredictAt = props.modelPrediction['predictAt'];
            const newPrediction = props.modelPrediction['prediction'];
            // Check if predictAt has changed by comparing objects
            const predictAtChanged = Object.keys(newPredictAt).length !== Object.keys(predictAt).length ||
                Object.keys(newPredictAt).some(key => newPredictAt[key] !== predictAt[key]);
            if (predictAtChanged) {
                setPredictAt(newPredictAt);
                setPrediction(newPrediction);
            }
        } else {
            setPredictAt({});
            setPrediction("");
        }
    }, [props.modelPrediction]);

    useEffect(() => {
        if (props.datasetFeatures) {
            if (!arrayEquals(datasetFeatures, props.datasetFeatures) || !(datasetResultParam === props.datasetResultParam)) {
                setDatasetFeatures(props.datasetFeatures);
                setDatasetFeaturesNo([...Array(props.datasetFeatures.length).keys()]);
                setDatasetResultParam(props.datasetResultParam);
                setPredictAt({});
                setPrediction("");
                // Initialize input values
                const initialValues = props.datasetFeatures.reduce(
                    (acc, feature) => ({ ...acc, [feature]: "" }),
                    {}
                );
                setInputValues(initialValues);
            }
        } else {
            setDatasetFeatures([]);
            setDatasetFeaturesNo([]);
            setDatasetResultParam("");
            setPredictAt({});
            setPrediction("");
            setInputValues({});
        }
    }, [props.datasetFeatures, props.datasetResultParam]);

    const handlePredict = () => {
        // Send dictionary with feature names as keys
        console.log('HANDLE PREDICT called, inputValues:', inputValues);
        if (props.onPredict) {
            props.onPredict(inputValues);
            console.log('onPredict called');
        }
    };

    const isPredicting = predictAt && typeof predictAt === 'object' && !Array.isArray(predictAt) && Object.keys(predictAt).length > 0 && !prediction;

    return (
        <div className="prediction-section">
            <h3>Prediction</h3>
            <div className="prediction-table-wrapper">
                <table className="prediction-table">
                    <tbody>
                        {datasetFeaturesNo.map((index) => {
                            const feature = datasetFeatures[index] || '';
                            const isNumber = index < (props.inputValidation?.noOfCts || noOfCts);
                            const featureOptions = props.inputValidation?.options?.[index - (props.inputValidation?.noOfCts || noOfCts)] || [];
                            const currentValue = inputValues[feature] || '';
                            
                            return (
                                <tr key={'pred-' + index}>
                                    <td className="prediction-feature-cell">{feature}</td>
                                    <td className="prediction-input-cell">
                                        {isNumber ? (
                                            <input
                                                type="number"
                                                step="any"
                                                className="prediction-input"
                                                value={currentValue}
                                                placeholder="Enter value"
                                                onChange={(e) => {
                                                    setInputValues({ ...inputValues, [feature]: e.target.value });
                                                }}
                                            />
                                        ) : (
                                            <select
                                                className="prediction-input"
                                                value={currentValue}
                                                onChange={(e) => {
                                                    setInputValues({ ...inputValues, [feature]: e.target.value });
                                                }}
                                            >
                                                <option value="">Select...</option>
                                                {featureOptions.map((option, i) => (
                                                    <option key={i} value={option}>{option}</option>
                                                ))}
                                            </select>
                                        )}
                                    </td>
                                </tr>
                            );
                        })}
                        <tr className="prediction-result-row">
                            <td className="prediction-feature-cell"><b>{datasetResultParam}</b></td>
                            <td className="prediction-result-cell">
                                {typeof prediction === 'string' || typeof prediction === 'number' ? prediction : '—'}
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
            <div className="prediction-actions">
                <button
                    className="ml-button"
                    onClick={handlePredict}
                    disabled={!datasetFeatures.length || isPredicting}
                >
                    {isPredicting ? 'Predicting...' : 'Predict'}
                </button>
            </div>
        </div>
    )

}

function getIndex(arr, value) {
    for(var i = 0; i < arr.length; i++) {
        if(arr[i] === value) {
            return i;
        }
    }
    return -1; //to handle the case where the value doesn't exist
}

function arrayEquals(a, b) {
    return Array.isArray(a) &&
        Array.isArray(b) &&
        a.length === b.length &&
        a.every((val, index) => val === b[index]);
}