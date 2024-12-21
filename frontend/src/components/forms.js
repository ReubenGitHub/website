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
            <label for="default">Use the default dataset <i>(vehicle emissions)</i></label>
            <br></br>
            <input type="radio" id="uploaded" name="datasetSelect" required onChange={(e) => setDatasetIsUpload(true)} ></input>
            <label for="uploaded">Upload a dataset (maximum size of 2MB)</label>
            <br></br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            { !datasetIsUpload && <input disabled type="file" /> }
            { datasetIsUpload && <input type="file" accept=".csv" id="dataset" onChange={(e) => setDataset(e.target.files[0])} /> }
            <br></br>
            <br></br>
            { (!datasetIsUpload || dataset) ?
                <button class="button-submit"> <b>Commit Dataset</b></button>:
                <button disabled class="button-submit"> <b>Commit Dataset</b></button>}
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
            <label for="supervised">Supervised</label>
            <input type="radio" disabled id="unsupervised" name="supervision" value="unsupervised" required onChange={(e) => setSupervision(e.target.value)} ></input>
            <label for="unsupervised">Unsupervised <i>(Coming soon...)</i></label>
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
                        <label for="regression">Regression</label>
                        <input type="radio" id="classification" name="resultType" value="classification" required onChange={(e) => {
                            setProblemType(e.target.value);
                            if (MLMethod !== "DT" && MLMethod !== "KNN") {
                                setMLMethod("DT");
                            }
                        }}></input>
                        <label for="classification">Classification</label>
                    </div>
                : (supervision==="unsupervised") ? 
                    <div>
                        <input type="radio" id="clustering" name="resultType" value="clustering" required onChange={(e) => setProblemType(e.target.value)}></input>
                        <label for="clustering">Clustering</label>
                        <input type="radio" id="association" name="resultType" value="association" required onChange={(e) => setProblemType(e.target.value)}></input>
                        <label for="association">Association</label>
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
            <br></br>
            <div class="grid-container">
                { datasetFieldsNo.map( (index) => (
                    <div class="grid-item">
                        {cateParams[index]===true}
                        { !(MLMethod==="PolyFit") ?
                                !(props.datasetFields['fields']) ?
                                    <input type="checkbox" disabled id={datasetFields['fields'][index]+"cts"} name="continuous features"></input>
                                : ( props.datasetFields['nonCtsFields'].includes(props.datasetFields['fields'][index]) ) ?
                                    <input type="checkbox" disabled id={datasetFields['fields'][index]+"cts"} name="continuous features" att="clearOnDataCommit" att3="clearOnPolyFit" ></input>
                                : <input type="checkbox" id={datasetFields['fields'][index]+"cts"} name="continuous features" att="clearOnDataCommit" att3="clearOnPolyFit" onChange={(e) =>  
                                            {ctsParams[index]=e.target.checked;
                                                cateParams[index]=false;
                                                document.querySelectorAll('input[id="'+datasetFields['fields'][index]+'ctg"]').forEach( el => el.checked = false )
                                            }
                                        }
                                    ></input>
                            : (MLMethod==="PolyFit") && 
                                !(props.datasetFields['fields']) ?
                                    <input type="radio" disabled id={datasetFields['fields'][index]+"cts"} name="continuous features"></input>
                                : ( props.datasetFields['nonCtsFields'].includes(props.datasetFields['fields'][index]) ) ?
                                    <input type="radio" disabled id={datasetFields['fields'][index]+"cts"} name="continuous features" att="clearOnDataCommit" att3="clearOnPolyFit" ></input>
                                : <input type="radio" id={datasetFields['fields'][index]+"cts"} name="continuous features" att="clearOnDataCommit" att3="clearOnPolyFit" onChange={(e) =>
                                        {var i;
                                            for (i=0; i<ctsParams.length; ++i) {
                                                ctsParams[i] = false;
                                            }
                                            ctsParams[index]=e.target.checked;
                                        }}
                                    ></input>
                        }
                        {(datasetFields['fields'][index].length>15) && <label for={datasetFields['fields'][index]+"cts"}>{datasetFields['fields'][index].substring(0,15)}... &nbsp;</label>}
                        {!(datasetFields['fields'][index].length>15) && <label for={datasetFields['fields'][index]+"cts"}>{datasetFields['fields'][index]} &nbsp;</label>}
                    </div>
                )) }
            </div>
            <br></br>
            <b>Features - Categorical</b>
            <br></br>
            <div class="grid-container">
                { datasetFieldsNo.map( (index) => (
                    <div class="grid-item">
                        {!(props.datasetFields['fields']) ?
                                <input type="checkbox" disabled id={datasetFields['fields'][index]+"ctg"} name="categorical features"></input>
                            : (MLMethod==="PolyFit") ?
                                <input type="checkbox" disabled id={datasetFields['fields'][index]+"ctg"} name="categorical features" att="clearOnDataCommit" att3="clearOnPolyFit"></input>
                            : <input type="checkbox" id={datasetFields['fields'][index]+"ctg"} name="categorical features" att="clearOnDataCommit" att3="clearOnPolyFit" onChange={(e) => 
                                        {cateParams[index]=e.target.checked;
                                            ctsParams[index]=false;
                                            document.querySelectorAll('input[id="'+datasetFields['fields'][index]+'cts"]').forEach( el => el.checked = false )
                                        }  
                                    }
                                ></input>
                        }
                        {(datasetFields['fields'][index].length>15) && <label for={datasetFields['fields'][index]+"ctg"}>{datasetFields['fields'][index].substring(0,15)}... &nbsp;</label>}
                        {!(datasetFields['fields'][index].length>15) && <label for={datasetFields['fields'][index]+"ctg"}>{datasetFields['fields'][index]} &nbsp;</label>}
                    </div>
                )) }
            </div>
            <br></br>
            <b>Result</b>
            <br></br>
            <div class="grid-container">
                { datasetFieldsNo.map( (index) => (
                    <div class="grid-item">
                        {!(props.datasetFields['fields']) ?
                                <input type="radio" disabled id={datasetFields['fields'][index]} name="result"></input>
                            :( (props.datasetFields['nonCtsFields'].includes(props.datasetFields['fields'][index])) && (problemType==="regression")  ) ?
                                <input type="radio" disabled id={datasetFields['fields'][index]} name="result" att="clearOnDataCommit" att2="clearOnRegression" required ></input>
                            :<input type="radio" id={datasetFields['fields'][index]} name="result" att="clearOnDataCommit" att2="clearOnRegression" required onChange={(e) => setResultParam(e.target.id) }></input>
                        }
                        {(datasetFields['fields'][index].length>15) && <label for={datasetFields['fields'][index]}>{datasetFields['fields'][index].substring(0,15)}... &nbsp;</label>}
                        {!(datasetFields['fields'][index].length>15) && <label for={datasetFields['fields'][index]}>{datasetFields['fields'][index]} &nbsp;</label>}
                    </div>
                )) }
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
            { props.isLoadingModelFit ? <button disabled class="button-submit"> <b>Fitting Model...</b></button>:
                !(props.datasetName) ? <button disabled class="button-submit"> <b>Fit Model</b></button>:
                <button class="button-submit"> <b>Fit Model</b></button>
            }  
        </form>
    )
    
}

export function FormPredictAt(props) {
    const [datasetFeatures, setDatasetFeatures] = useState(["Features..."]);
    const [datasetFeaturesNo, setDatasetFeaturesNo] = useState(["0"])
    const [predictAt, setPredictAt] = useState([""]);
    const [predictValues, setPredictValues] = useState([""]);
    const [predictEnable, setPredictEnable] = useState(0);
    // const validOptions = props.inputValidation['options'];
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
                setPredictValues( Array(props.inputValidation['features'].length).fill("") );
                document.querySelectorAll('input[att=clearOnFeatureSelect],select[att=clearOnFeatureSelect]').forEach( el => el.value = "" );
            }
        } else {
            setInputValidation({
                features: ["Features..."],
                noOfFeatures: [...Array( 1).keys()],   //For checkboxes table setup
                noOfCts: 0,
                options: [["Option"]]
            });
            setPredictValues( Array(1).fill("") );
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
                { (inputValidation['noOfFeatures']).map( (index) => (
                    <tr>
                        <td align="right">
                            <label for={inputValidation['features'][index]}> {inputValidation['features'][index]} </label>
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
                                        onChange={(e) => predictValues[index] = e.target.value }
                                        />
                                    :<select id={inputValidation['features'][index]}
                                        att="clearOnFeatureSelect"
                                        style={{width: "100%"}}
                                        required
                                        onChange={(e) => predictValues[index] = e.target.value} >
                                        <option value="" selected></option>
                                        { (inputValidation['options'][(index-inputValidation['noOfCts'])]).map( (option) => (
                                            <option value={option}>{option}</option>
                                        )) }
                                        </select>
                                }
                        </td>
                    </tr>
                )) }
            </table>

            <br></br>
            { props.isLoadingModelPredict ? <button disabled class="button-submit"> <b>Predicting {props.datasetResultParam}...</b></button>:
                props.isLoadingModelFit ? <button disabled class="button-submit"> <b>Predict</b></button>:
                !(props.predictionTitle) ? <button disabled class="button-submit"> <b>Predict</b></button>:
                <button class="button-submit"> <b>Predict {props.datasetResultParam}</b></button>
            } 
        </form>
    )
}
// selected={getIndex(validOptions[(index-props.noOfCtsParams)], option)===3 }

export function FormModelOutputs(props) {
    const modelOutputs = props.modelOutputs
    const accuracyTrain = modelOutputs['accuracyTrain']
    const accuracyTest = modelOutputs['accuracyTest']
    const precsRecs = modelOutputs['precsRecs']
    const repImageCreated = modelOutputs['repImageCreated']
    const repImageBase64 = modelOutputs['repImageBase64']

    const repImageSrc = !accuracyTrain ? representationIcon
        : !repImageCreated ? representationNAIcon
        : `data:image/png;base64,${repImageBase64}`

    return (
        <div>
            
            <br></br>
            <img class="img-center" src={repImageSrc} alt="Model Representation" width="80%" border="1px"></img>
            <br></br>
            <hr color="#03bffe"></hr>
            <h3 align="center">Model Metrics</h3>
            <br></br>
            <table align="center">
                <tr >
                    <td style={{padding: "4px 8px"}}>
                    </td>
                    {(precsRecs) ?
                            <th style={{padding: "4px 8px", background: "#ffa78a"}}>
                                Classifier
                                <br></br>
                                Accuracy
                            </th>:
                        (accuracyTrain) ?
                            <th style={{padding: "4px 8px", background: "#ffa78a"}}>
                                R-Squared
                                <br></br>
                                Accuracy
                            </th>:
                        <th style={{padding: "4px 8px", background: "#ffa78a"}}>
                            Accuracy
                        </th>
                    }
                    {(precsRecs) && 
                        <th style={{padding: "4px 8px", background: "#ffa78a"}}>
                            Macro | Micro
                            <br></br>
                            Precision
                        </th>
                    }
                    {(precsRecs) && 
                        <th style={{padding: "4px 8px", background: "#ffa78a"}}>
                            Macro | Micro
                            <br></br>
                            Recall
                        </th>
                    }
                </tr>
                <tr >
                    <td align="left" style={{padding: "4px 8px", background: "#ffa78a"}}>
                        <b>Training </b>
                    </td>
                    <td align="center" style={{padding: "4px 8px", background: "#ffcab8"}}>
                        { !(accuracyTrain==null) ? 
                            ( 1*accuracyTrain + Number.EPSILON ).toFixed(3) :
                            "..."
                        }
                    </td>
                    {(precsRecs) && 
                        <td align="center" style={{padding: "4px 8px", background: "#ffcab8"}}>
                            <div> {( 1*precsRecs['macroPrecTrain'] + Number.EPSILON ).toFixed(3)} &nbsp;|&nbsp; {( 1*precsRecs['microPrecTrain'] + Number.EPSILON ).toFixed(3)} </div>
                        </td>
                    }
                    {(precsRecs) &&
                        <td align="center" style={{padding: "4px 8px", background: "#ffcab8"}}>
                            <div> {( 1*precsRecs['macroRecallTrain'] + Number.EPSILON ).toFixed(3)} &nbsp;|&nbsp; {( 1*precsRecs['microRecallTrain'] + Number.EPSILON ).toFixed(3)} </div>
                        </td>
                    }
                </tr>
                <tr >
                    <td align="left" style={{padding: "4px 8px", background: "#ffa78a"}}>
                        <b>Testing </b>
                    </td>
                    <td align="center" style={{padding: "4px 8px", background: "#ffcab8"}}>
                        { !(accuracyTest==null) ? 
                            ( 1*accuracyTest + Number.EPSILON ).toFixed(3) :
                            "..."
                        }
                    </td>
                    {(precsRecs) && 
                        <td align="center" style={{padding: "4px 8px", background: "#ffcab8"}}>
                            { (precsRecs) ? 
                                <div> {( 1*precsRecs['macroPrecTest'] + Number.EPSILON ).toFixed(3)} &nbsp;|&nbsp; {( 1*precsRecs['microPrecTest'] + Number.EPSILON ).toFixed(3)} </div> :
                                "..."
                            }
                        </td>
                    }
                    {(precsRecs) && 
                        <td align="center" style={{padding: "4px 8px", background: "#ffcab8"}}>
                            { (precsRecs) ? 
                                <div> {( 1*precsRecs['macroRecallTest'] + Number.EPSILON ).toFixed(3)} &nbsp;|&nbsp; {( 1*precsRecs['microRecallTest'] + Number.EPSILON ).toFixed(3)} </div> :
                                "..."
                            }
                        </td>
                    }
                </tr>
            </table>
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
            <h3 align="center">Prediction</h3>
            <br></br>
            <table align="center">
                { (datasetFeaturesNo).map( (index) => (
                    <tr  >
                        { (index%2) ?
                            <td align="left" style={{padding: "4px 8px", background: "#abe3ff"}}>
                                {datasetFeatures[index]}
                            </td>:
                            <td align="left" style={{padding: "4px 8px", background: "#abe3ff"}}>
                                {datasetFeatures[index]}
                            </td>
                        }
                        { (index%2) ?
                            <td align="center" style={{padding: "4px 8px", background: "#def4ff"}}>
                                { (Array.isArray(predictAt)) ? 
                                    predictAt[index] :
                                    "..."
                                }
                            </td>:
                            <td align="center" style={{padding: "4px 8px", background: "#def4ff"}}>
                                { (Array.isArray(predictAt)) ? 
                                    predictAt[index] :
                                    "..."
                                }
                            </td>
                        }
                    </tr>
                )) }
                <tr>
                    <td align="left" style={{padding: "4px 8px", background: "#ffa78a"}}>
                        <b>{datasetResultParam}</b>
                    </td>
                    <td align="center" style={{padding: "4px 8px", background: "#ffcab8"}}>
                        { (prediction) ? 
                            prediction :
                            "..."
                        }
                    </td>
                </tr>
            </table>
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