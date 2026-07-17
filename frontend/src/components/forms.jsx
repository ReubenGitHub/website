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
            <br></br>
            <input type="radio" id="default" name="datasetSelect" required defaultChecked onChange={(e) => {setDatasetIsUpload(false); setDataset()} } ></input>
            <label htmlFor="default">Use the default dataset <i>(vehicle emissions)</i></label>
            <br></br>
            <input type="radio" id="uploaded" name="datasetSelect" required onChange={(e) => setDatasetIsUpload(true)} ></input>
            <label htmlFor="uploaded">Upload a dataset (maximum size of 2MB)</label>
            <br></br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            { !datasetIsUpload && <input disabled type="file" /> }
            { datasetIsUpload && <input type="file" accept=".csv" id="dataset" onChange={(e) => setDataset(e.target.files[0])} /> }
            <br></br>
            <br></br>
            { (!datasetIsUpload || dataset) ?
                <button className="ml-button">Commit Dataset</button> :
                <button disabled className="ml-button">Commit Dataset</button> }
            
            {datasetName && (
                <div className="dataset-badge">
                    <span className="dataset-badge-icon">📄</span>
                    <span>{datasetName}</span>
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
    const [supervision, setSupervision] = useState("supervised")
    
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
            setInputs([supervision, problemType, MLMethod, Number(PolyDeg), ctsFeatures, cateFeatures, resultParam, testProp/100]) //Asynchronous, just starts a queue so small delay, use below vv
        }
        // setInputs([supervision, problemType, MLMethod, Number(PolyDeg), ctsFeatures, cateFeatures, resultParam, testProp/100]) //Asynchronous, just starts a queue so small delay, use below vv
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
            <b>Supervision</b>
            <br></br>
            <input type="radio" id="supervised" name="supervision" checked value="supervised" required onChange={(e) => setSupervision(e.target.value)} ></input>
            <label htmlFor="supervised">Supervised</label>
            <input type="radio" disabled id="unsupervised" name="supervision" value="unsupervised" required onChange={(e) => setSupervision(e.target.value)} ></input>
            <label htmlFor="unsupervised">Unsupervised <i>(Coming soon...)</i></label>
            <br></br>
            <br></br>
            <b>Problem Type</b>
            <br></br>
            { (supervision==="supervised") ?
                    <div>
                        <input type="radio" id="regression" name="resultType" value="regression" required onChange={(e) => {
                            setProblemType(e.target.value);
                            { ( props.datasetFields['nonCtsFields'].includes(resultParam) ) && setResultParam("") }
                            { ( props.datasetFields['nonCtsFields'].includes(resultParam) ) && document.querySelectorAll('input[att2=clearOnRegression]').forEach( el => el.checked = false ) }
                        }}></input>
                        <label htmlFor="regression">Regression</label>
                        <input type="radio" id="classification" name="resultType" value="classification" required onChange={(e) => {
                            setProblemType(e.target.value);
                            if (MLMethod !== "DT" && MLMethod !== "KNN") {
                                setMLMethod("DT");
                            }
                        }}></input>
                        <label htmlFor="classification">Classification</label>
                    </div>
                : (supervision==="unsupervised") ? 
                    <div>
                        <input type="radio" id="clustering" name="resultType" value="clustering" required onChange={(e) => setProblemType(e.target.value)}></input>
                        <label htmlFor="clustering">Clustering</label>
                        <input type="radio" id="association" name="resultType" value="association" required onChange={(e) => setProblemType(e.target.value)}></input>
                        <label htmlFor="association">Association</label>
                    </div>
                : <div>Please specifiy a supervision...</div>
            }
            <br></br>
            <b>Machine Learning Method</b>
            <br></br>
            <label> Model type &nbsp;
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
            </label>
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
                                    if (isPolyFit) {
                                        var i;
                                        for (i=0; i<ctsParams.length; ++i) {
                                            ctsParams[i] = false;
                                        }
                                        ctsParams[index] = e.target.checked;
                                    } else {
                                        ctsParams[index] = e.target.checked;
                                        cateParams[index] = false;
                                        document.querySelectorAll('input[id="'+fieldName+'ctg"]').forEach( el => el.checked = false );
                                    }
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
            <label> Percentage &nbsp;
                <input
                    type="number"
                    min="1"
                    max="99"
                    value={testProp}
                    required
                    onChange={(e) => setTestProp(e.target.value)}
                />
            </label>
            %
            <br></br>
            <br></br>
            { props.isLoadingModelFit ? <button disabled className="ml-button">Fitting Model...</button>:
                !(props.datasetName) ? <button disabled className="ml-button">Fit Model</button>:
                <button className="ml-button">Fit Model</button>
            }  
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
        <div className="ml-results-layout">
            <div className="ml-results-image">
                <img src={repImageSrc} alt="Model Representation" />
            </div>
            <div className="ml-results-metrics">
                <h3>Model Metrics</h3>
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
    const [datasetFeatures, setDatasetFeatures] = useState(["Features..."]);
    const [datasetResultParam, setDatasetResultParam] = useState(["Result..."])
    const [datasetFeaturesNo, setDatasetFeaturesNo] = useState(["0"]);
    const [predictAt, setPredictAt] = useState("");
    const [prediction, setPrediction] = useState("");

    useEffect(() => {
        if (props.modelPrediction['predictAt']) {
            if (!arrayEquals(predictAt,props.modelPrediction['predictAt'])) {
                setPredictAt(props.modelPrediction['predictAt']);
                setPrediction( props.modelPrediction['prediction'] );
            }
        } else {
            setPredictAt("");
            setPrediction( "" );
        }
    }, [props.modelPrediction]);

    useEffect(() => {
        if (props.datasetFeatures) {
            if (!arrayEquals(datasetFeatures,props.datasetFeatures) | !(datasetResultParam===props.datasetResultParam)) {
                setDatasetFeatures(props.datasetFeatures);
                setDatasetFeaturesNo( [...Array( props.datasetFeatures.length).keys()] ); //For checkboxes table setup
                setDatasetResultParam( props.datasetResultParam );
                setPredictAt("");
                setPrediction( "" );
            }
        } else {
            setDatasetFeatures(["Features..."]);
            setDatasetFeaturesNo( [...Array( 1).keys()] ); //For checkboxes table setup
            setDatasetResultParam( ["Result..." ]);
            setPredictAt("");
            setPrediction( "" );
        }
    }, [props.datasetFeatures, props.datasetResultParam]);


    return (
        <div>
            <h3>Prediction</h3>
            <div className="prediction-list">
                { (Array.isArray(predictAt)) ? predictAt.map( (val, index) => (
                    <div key={'pred-'+index} className="prediction-row">
                        <span className="prediction-feature">{datasetFeatures[index]}</span>
                        <span className="prediction-value">{val}</span>
                    </div>
                )) : (
                    <div className="prediction-row">
                        <span className="prediction-feature">Loading...</span>
                        <span className="prediction-value">...</span>
                    </div>
                )}
                <div className="prediction-row prediction-result">
                    <span className="prediction-feature"><b>{datasetResultParam}</b></span>
                    <span className="prediction-value">
                        {prediction ? prediction : "..."}
                    </span>
                </div>
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