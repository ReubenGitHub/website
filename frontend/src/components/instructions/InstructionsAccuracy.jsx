import React from 'react';
import '../machinelearner.css';

export function InstructionsAccuracy() {
    return (
        <div>
            <div className="info-section">
                <h4>Model Accuracy</h4>
                <p>A measure of how well your model predicts results, assessed by comparing model predictions with the true results given in the data.</p>
                <p><i>R2 accuracy</i>, also known as the <i>Coefficient of Determination</i>, measures the accuracy of regression models. The best possible score is 1.0. If the score is 0.0, a constant model which always predicts the mean value of the result values would be just as accurate as your model.</p>
                <p><i>Classifier accuracy</i> measures the accuracy of classification models. It is the proportion of predictions which are correct. The best possible score is 1.0.</p>
            </div>
            <div className="info-section">
                <h4>Data Leakage</h4>
                <p>If your model has a surprisingly high accuracy on both training and testing data, this could be a sign of <i>data leakage</i>, which means your model has used information while predicting that would not normally be available when making predictions in the real world.</p>
                <p>One form of data leakage is <i>Feature Leakage</i>, where information about the result is leaked into the selected features, for example, including an "hoursAwake" feature when trying to predict "hoursAsleep".</p>
                <p>Another form of data leakage is <i>Training Data Leakage</i>, where information is shared between entries in the dataset, meaning your model gets a sneak-peak at the testing data while training. This can happen if there are duplicate entries in the data, or if entries aren't i.i.d.</p>
            </div>
            <div className="info-section">
                <h4>Imbalanced Data</h4>
                <p>Surprisingly high accuracies can also be a sign of <i>Imbalanced Data</i>, meaning your data has a minority class in the result. For example, if the result is "Red" 99% of the time, your model might learn to constantly predict "Red" to achieve 99% accuracy.</p>
                <p>While correct almost all of the time, your model will never identify rare results, so accuracy is not the be-all and end-all in assessing models. To improve models on imbalanced data, one can consider downsampling and upweighting larger classes. This is not yet performed in this app.</p>
            </div>
            <div className="info-section">
                <h4>Over-Fitting</h4>
                <p>If your model has high accuracy on training data, but low accuracy on testing data, this might indicate <i>over-fitting</i>, whereby your model is overly-complex and hyper-specified to perform well on your training data, but fails to generalise to unseen training data.</p>
            </div>
            <div className="info-section">
                <h4>Bias</h4>
                <p>If your model has low accuracy on both training and testing data, your model might be <i>biased</i> or might not be complex enough. Consider providing your model with more (relevant) features.</p>
            </div>
        </div>
    );
}
