import './pagestyles.css';
import './home.css';

export default function MlVehicleEmissionsPage() {
    return (
        <div className="column-container">
            <section className="section">
                <div className="card">
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
                    <a
                        href="https://github.com/ReubenGitHub/ML-Vehicle-Emissions"
                        className="project-link external-link"
                        target="_blank"
                        rel="noreferrer"
                    >
                        View on GitHub
                    </a>
                </div>
            </section>
        </div>
    );
}
