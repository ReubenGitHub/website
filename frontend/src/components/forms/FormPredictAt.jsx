import React, {useState, useEffect} from 'react';
import './forms.css';

function arrayEquals(a, b) {
    return Array.isArray(a) &&
        Array.isArray(b) &&
        a.length === b.length &&
        a.every((val, index) => val === b[index]);
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
        setPredictAt( predictValues);
        setPredictEnable( predictEnable+1 );
    }

    useEffect(() => {
        if (props.inputValidation['features']) {
            if (!arrayEquals(inputValidation,props.inputValidation)) {
                setInputValidation({
                    features: props.inputValidation['features'],
                    noOfFeatures: [...Array( props.inputValidation['features'].length).keys()],
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
                noOfFeatures: [...Array( 1).keys()],
                noOfCts: 0,
                options: [["Option"]]
            });
            setPredictValues({});
            document.querySelectorAll('input[att=clearOnFeatureSelect],select[att=clearOnFeatureSelect]').forEach( el => el.value = "" );
        }
    }, [props.inputValidation]);

    useEffect(() => {
        if (predictEnable>0) {
            props.parentCallback(predictAt)
        }
        setPredictEnable( 0 );
    }, [predictEnable]);

    const TooltipIcon = ({ text }) => (
        <span className="tooltip-wrapper">
            <span className="tooltip-icon">ℹ</span>
            <span className="tooltip-text">{text}</span>
        </span>
    );

    return (
        <form onSubmit={handleSubmit}>
            <b>Predict a result at <TooltipIcon text="Enter values for each feature to get a prediction. Numerical fields accept numbers. Categorical fields have a dropdown of available options." /></b>
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
                <button className="ml-button ml-button-predict">Predict {props.datasetResultParam} <TooltipIcon text="Click to generate a prediction using your trained model and the feature values entered above." /></button>
            } 
        </form>
    )
}
