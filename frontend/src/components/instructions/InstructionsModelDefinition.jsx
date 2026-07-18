import React from 'react';
import '../machinelearner.css';

export function InstructionsModelDefinition() {
    return (
        <div>
            <div className="info-section">
                <h4>Problem Type</h4>
                <p>Select <i>Regression</i> if the result you want to predict is a continuous variable, i.e. numerical and where some values can be "bigger" than others.</p>
                <p>Select <i>Classification</i> if the result you want to predict is a categorical variable, i.e. categories or types of something, with no sense of "bigger" or "smaller".</p>
            </div>
            <div className="info-section">
                <h4>Machine Learning Method</h4>
                <p><i>Decision Trees</i> will derive a prediction through a sequence of decisions based on known features, similar to the game '20Q'.</p>
                <p><i>K-Nearest Neighbours</i> predicts a result by averaging the results of K training samples which have feature values closest to the feature values to predict with.
                If the result is continuous, the result is the mean of nearby neighbours. If the result is categorical, the result is the mode (most common) of nearby neighbours.
                "Closeness" of samples is determined by the Euclidean distance on continuous features, plus the Hamming distance on categorical features.</p>
                <p><i>Linear Regression</i> assumes that the result is a linear combination of the features (and that features are independent of one another).
                For continuous features this means determining a line of best fit for one feature, or a plane of best fit for two features, and so on.
                For categorical features, the model will determine the best constant to add to the prediction for each possible category.</p>
                <p><i>Polynomial Regression</i> assumes that the result is some polynomial of a single continuous feature. The model determines the best-fitting polynomial of the specified degree.</p>
            </div>
        </div>
    );
}
