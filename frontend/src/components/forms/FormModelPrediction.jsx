import React, {useState, useEffect} from 'react';
import './forms.css';

function arrayEquals(a, b) {
    return Array.isArray(a) &&
        Array.isArray(b) &&
        a.length === b.length &&
        a.every((val, index) => val === b[index]);
}

export function FormModelPrediction(props) {
    const [datasetFeatures, setDatasetFeatures] = useState([]);
    const [datasetResultParam, setDatasetResultParam] = useState("");
    const [datasetFeaturesNo, setDatasetFeaturesNo] = useState([]);
    const [predictAt, setPredictAt] = useState({});
    const [prediction, setPrediction] = useState("");
    const [inputValues, setInputValues] = useState({});
    const [noOfCts, setNoOfCts] = useState(0);
    const [options, setOptions] = useState([[]]);

    const TooltipIcon = ({ text }) => (
        <span className="tooltip-wrapper">
            <span className="tooltip-icon">ℹ</span>
            <span className="tooltip-text">{text}</span>
        </span>
    );

    useEffect(() => {
        if (props.modelPrediction && props.modelPrediction['predictAt'] && typeof props.modelPrediction['predictAt'] === 'object' && !Array.isArray(props.modelPrediction['predictAt']) && Object.keys(props.modelPrediction['predictAt']).length > 0) {
            const newPredictAt = props.modelPrediction['predictAt'];
            const newPrediction = props.modelPrediction['prediction'];
            // Check if predictAt OR prediction has changed
            const predictAtChanged = Object.keys(newPredictAt).length !== Object.keys(predictAt).length ||
                Object.keys(newPredictAt).some(key => newPredictAt[key] !== predictAt[key]);
            const predictionChanged = newPrediction !== prediction;
            if (predictAtChanged || predictionChanged) {
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
        if (props.onPredict) {
            props.onPredict(inputValues);
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
                                <span className="prediction-value-wrapper">
                                    {typeof prediction === 'number' ? parseFloat(prediction.toPrecision(6)) : (typeof prediction === 'string' ? prediction : '—')}
                                    <TooltipIcon text="This is the model's predicted value for your input features. It is generated from the trained model using your provided data." />
                                </span>
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
