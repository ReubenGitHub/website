import '../global.css';
import '../headernavbar.css';
import '../Hero.css';
import '../home.css';

export default function MlVehicleEmissionsPage() {
    return (
        <div className="column-container">
            <section className="section">
                <div className="card project-page-card">
                    <h2>Machine Learning Vehicle Emissions</h2>
                    <p className="intro">
                        An investigation into UK vehicle CO2 emissions using machine learning. 
                        I sourced, processed, and analysed UK government vehicle emissions data 
                        for 6,756 vehicles, building predictive models to estimate CO2 emissions 
                        from vehicle features like powertrain type and engine size.
                    </p>
                    <p className="intro">
                        The project involved exploratory data analysis using MySQL, Pandas, NumPy, 
                        Matplotlib, and Seaborn. Hyperparameters were tuned using grid search, 
                        random search, and Bayesian optimisation (Trieste and Hyperopt). XGBoost 
                        models were compared and assessed through feature importance analysis 
                        and prediction error distributions.
                    </p>
                    <div className="project-images">
                        <div className="image-caption-group">
                            <img 
                                src="/assets/projects/ml-emissions/co2-vs-power.png" 
                                alt="CO2 Emissions vs Power" 
                                className="project-image"
                            />
                            <span className="image-caption">CO2 Emissions vs Power: Train/test split analysis showing emissions distribution across vehicle power</span>
                        </div>
                        <div className="image-caption-group">
                            <img 
                                src="/assets/projects/ml-emissions/prediction-error.png" 
                                alt="Prediction Error Distribution" 
                                className="project-image"
                            />
                            <span className="image-caption">Prediction Error Distribution (Full Model): Model errors across percentiles of the data</span>
                        </div>
                    </div>
                    <a
                        href="https://github.com/ReubenGitHub/ML-Vehicle-Emissions"
                        className="project-link external-link"
                        target="_blank"
                        rel="noreferrer"
                    >
                        View on GitHub
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 -960 960 960" fill="currentColor" style={{verticalAlign: 'middle', marginLeft: 4}}><path d="M200-120q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h280v80H200v560h560v-280h80v280q0 33-23.5 56.5T760-120H200Zm188-212-56-56 372-372H560v-80h280v280h-80v-144L388-332Z"/></svg>
                    </a>
                </div>
            </section>
        </div>
    );
}
