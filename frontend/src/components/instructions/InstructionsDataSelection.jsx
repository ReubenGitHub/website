import React from 'react';
import '../machinelearner.css';

export function InstructionsDataSelection() {
    return (
        <div>
            <div className="info-section">
                <h4>Upload Your Data</h4>
                <p>Upload any CSV dataset where rows are entries, columns are fields, and column headers are included.</p>
            </div>
            <div className="info-section">
                <h4>Default Dataset</h4>
                <p>Select the default dataset, which contains vehicle emissions data such as make, model, engine size, and CO2 emissions (g/km).</p>
                <p>The raw data was obtained from the UK government's vehicle certification agency at <a href="https://www.gov.uk/governmentorganisations/vehicle-certification-agency" target="_blank" rel="noopener noreferrer">https://www.gov.uk/governmentorganisations/vehicle-certification-agency</a>. I then cleaned this data as part of my ML investigation into CO2 emissions at <a href="https://github.com/ReubenGitHub/ML-Vehicle-Emissions" target="_blank" rel="noopener noreferrer">https://github.com/ReubenGitHub/ML-Vehicle-Emissions</a>.</p>
            </div>
        </div>
    );
}
