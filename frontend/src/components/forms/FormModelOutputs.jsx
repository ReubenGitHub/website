import React from 'react';
import './forms.css';
import representationIcon from '../../assets/images/repres_icon.png';
import representationNAIcon from '../../assets/images/repres_na_icon.png';

export function FormModelOutputs(props) {
    const { modelMetrics, graphImageBase64 } = props.modelOutputs

    const repImageSrc = !modelMetrics?.train_accuracy
        ? representationIcon
        : !graphImageBase64
            ? representationNAIcon
            : `data:image/png;base64,${graphImageBase64}`

    const isClassification = !!modelMetrics?.train_macro_precision;
    const accuracyLabel = isClassification ? 'Classifier Accuracy' : 'R-Squared Accuracy';

    const TooltipIcon = ({ text }) => (
        <span className="tooltip-wrapper">
            <span className="tooltip-icon">ℹ</span>
            <span className="tooltip-text">{text}</span>
        </span>
    );

    return (
        <div className="ml-results-container">
            <div className="ml-results-image">
                <img src={repImageSrc} alt="Model Representation" />
            </div>
            <div className="ml-results-metrics">
                <div className="metric-card">
                    {/* Accuracy Section */}
                    <div className="metric-section">
                        <div className="metric-card-header">{accuracyLabel} <TooltipIcon text={isClassification ? 'Accuracy: Percentage of correct predictions. Higher is better, with 100% being perfect classification.' : 'R-Squared: Proportion of variance explained by the model. Ranges from 0 to 1, where 1 is a perfect fit.'} /></div>
                        <div className="metric-card-row">
                            <span className="metric-card-label">Train</span>
                            <span className="metric-card-value">
                                {!(modelMetrics?.train_accuracy == null) ?
                                    ((1 * modelMetrics?.train_accuracy + Number.EPSILON) * 100).toPrecision(3) + '%'
                                    : '...'}
                            </span>
                            <span className="metric-divider"></span>
                            <span className="metric-card-label">Test</span>
                            <span className="metric-card-value">
                                {!(modelMetrics?.test_accuracy == null) ?
                                    ((1 * modelMetrics?.test_accuracy + Number.EPSILON) * 100).toPrecision(3) + '%'
                                    : '...'}
                            </span>
                        </div>
                    </div>

                    {/* Precision Section (Classification only) */}
                    {isClassification && (
                        <div className="metric-section">
                            <div className="metric-card-header">Precision <TooltipIcon text="Precision: Of all predicted positives, how many were actually positive? Macro averages metrics equally across classes. Micro calculates globally across all classes." /></div>
                            <div className="metric-card-row">
                                <span className="metric-card-label">Train Macro</span>
                                <span className="metric-card-value">
                                    {((1 * modelMetrics?.train_macro_precision + Number.EPSILON) * 100).toPrecision(3) + '%'}
                                </span>
                                <span className="metric-divider"></span>
                                <span className="metric-card-label">Train Micro</span>
                                <span className="metric-card-value">
                                    {((1 * modelMetrics?.train_micro_precision + Number.EPSILON) * 100).toPrecision(3) + '%'}
                                </span>
                            </div>
                            <div className="metric-card-row">
                                <span className="metric-card-label">Test Macro</span>
                                <span className="metric-card-value">
                                    {((1 * modelMetrics?.test_macro_precision + Number.EPSILON) * 100).toPrecision(3) + '%'}
                                </span>
                                <span className="metric-divider"></span>
                                <span className="metric-card-label">Test Micro</span>
                                <span className="metric-card-value">
                                    {((1 * modelMetrics?.test_micro_precision + Number.EPSILON) * 100).toPrecision(3) + '%'}
                                </span>
                            </div>
                        </div>
                    )}

                    {/* Recall Section (Classification only) */}
                    {isClassification && (
                        <div className="metric-section">
                            <div className="metric-card-header">Recall <TooltipIcon text="Recall (Sensitivity): Of all actual positives, how many did the model correctly predict? Higher recall means fewer false negatives." /></div>
                            <div className="metric-card-row">
                                <span className="metric-card-label">Train Macro</span>
                                <span className="metric-card-value">
                                    {((1 * modelMetrics?.train_macro_recall + Number.EPSILON) * 100).toPrecision(3) + '%'}
                                </span>
                                <span className="metric-divider"></span>
                                <span className="metric-card-label">Train Micro</span>
                                <span className="metric-card-value">
                                    {((1 * modelMetrics?.train_micro_recall + Number.EPSILON) * 100).toPrecision(3) + '%'}
                                </span>
                            </div>
                            <div className="metric-card-row">
                                <span className="metric-card-label">Test Macro</span>
                                <span className="metric-card-value">
                                    {((1 * modelMetrics?.test_macro_recall + Number.EPSILON) * 100).toPrecision(3) + '%'}
                                </span>
                                <span className="metric-divider"></span>
                                <span className="metric-card-label">Test Micro</span>
                                <span className="metric-card-value">
                                    {((1 * modelMetrics?.test_micro_recall + Number.EPSILON) * 100).toPrecision(3) + '%'}
                                </span>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}
