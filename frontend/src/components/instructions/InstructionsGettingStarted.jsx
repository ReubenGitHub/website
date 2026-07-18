import React from 'react';
import '../machinelearner.css';

export function InstructionsGettingStarted() {
    return (
        <div>
            <p className="info-text">
                This app is a general machine learner: designed to be intuitive, and built to allow you to fit a machine-led model to any data you like!
            </p>
            <p className="info-text">
                The stages involved in using this app are: choosing a dataset, configuring model parameters, selecting features and the result to predict, reviewing the model representation and accuracy, and using your model to predict results.
            </p>
            <p className="info-text">
                Click through these tabs for instructions to help you through each stage of building your models. Alternatively, get started below using the tooltip icons for help!
            </p>
        </div>
    );
}
