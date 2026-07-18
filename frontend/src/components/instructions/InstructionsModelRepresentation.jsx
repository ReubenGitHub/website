import React from 'react';
import '../machinelearner.css';

export function InstructionsModelRepresentation() {
    return (
        <div>
            <div className="info-section">
                <h4>Model Map</h4>
                <p>A visual representation of your model. Representations are generated for Decision Trees, KNN with one or two continuous features, Linear Regression models with one or two continuous features, and Polynomial Regression models.</p>
                <p>Representations of Decision Trees are reduced down to a maximum depth of 4 for enhanced visibility, regardless of the actual maximum depth of the model.</p>
            </div>
        </div>
    );
}
