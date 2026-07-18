import React, {useState, useEffect} from 'react';
import './forms.css';

function arrayEquals(a, b) {
    return Array.isArray(a) &&
        Array.isArray(b) &&
        a.length === b.length &&
        a.every((val, index) => val === b[index]);
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
    
    const TooltipIcon = ({ text }) => (
        <span className="tooltip-wrapper">
            <span className="tooltip-icon">ℹ</span>
            <span className="tooltip-text">{text}</span>
        </span>
    );
    
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
            // Button is disabled so this won't happen
        } else {
            setInputs([problemType, MLMethod, Number(PolyDeg), ctsFeatures, cateFeatures, resultParam, testProp/100])
        }
    }
    
    const hasFeatures = datasetFields['fields'].some((_, i) => ctsParams[i] || cateParams[i]);
    const canFitModel = problemType && resultParam && hasFeatures && !props.isLoadingModelFit;
    
    const requiresProblemType = !problemType;
    const requiresResult = !resultParam;
    const requiresFeature = !hasFeatures;
    
    const ChecklistItem = ({ met, text }) => (
        <span className="ml-checklist-item">
            <span className={`ml-check-icon ${met ? 'ml-check-met' : 'ml-check-unmet'}`}>{met ? '✓' : '○'}</span>
            <span className={met ? 'ml-check-text-met' : 'ml-check-text-unmet'}>{text}</span>
        </span>
    );

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
            <b>Problem Type <TooltipIcon text="Select Regression for continuous numerical results (e.g., predicting CO2 emissions, temperature). Select Classification for categories or classes (e.g., spam/ham, disease/no disease)." /></b>
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
            <b>Model Type <TooltipIcon text="Decision Trees: sequence of decisions like '20 Questions'. K-Nearest Neighbours: classifies based on closest data points. Linear Regression: fits a straight line. Polynomial Regression: fits a curved line." /></b>
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
            <b>Features - Continuous <TooltipIcon text="Select fields with numerical data that can be measured or counted (e.g., age, temperature, distance, income)." /></b>
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
            <b>Features - Categorical <TooltipIcon text="Select fields with categories or options (e.g., color: red/blue/green, type: A/B/C, yes/no). Each value represents a group, not a number." /></b>
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
            <b>Result <TooltipIcon text="The field you want the model to predict or classify. This is your target variable. For regression, choose a numerical field. For classification, choose a categorical field." /></b>
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
            <b>Test Data Proportion <TooltipIcon text="Percentage of your dataset set aside for testing the model's accuracy. The model never sees this data during training. Typical values: 20-30%. Higher values = more reliable testing but less data for training." /></b>
            <br></br>
            <div className="test-proportion-wrapper">
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
            </div>
            <br></br>
            <div className="ml-checklist">
                <ChecklistItem met={!requiresProblemType} text="Select a problem type" />
                <ChecklistItem met={!requiresFeature} text="Select at least one feature" />
                <ChecklistItem met={!requiresResult} text="Select a result" />
            </div>
            <div className="ml-button-container">
                { props.isLoadingModelFit ? <button disabled className="ml-button">Fitting Model...</button>:
                    !canFitModel ? <button disabled className="ml-button">Fit Model</button>:
                    <button className="ml-button">Fit Model</button>
                }
            </div>  
        </form>
    )
    
}
