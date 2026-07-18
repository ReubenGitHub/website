import React from 'react';
import '../machinelearner.css';

export function InstructionsFeaturesResult() {
    return (
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
    );
}
