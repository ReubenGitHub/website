import React from 'react';
import '../machinelearner.css';

export function InstructionsPrediction() {
    return (
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
    );
}
