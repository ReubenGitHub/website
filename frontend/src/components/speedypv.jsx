import './pagestyles.css';
import './home.css';

export default function SpeedyPVPage() {
    return (
        <div className="column-container">
            <section className="section">
                <div className="card">
                    <h2>Speedy PV</h2>
                    <p className="intro">
                        A solar PV lead generator built while working at Midsummer Energy. 
                        This tool helps generate quotes and leads for solar panel installations, 
                        streamlining the sales process for solar PV projects.
                    </p>
                    <a
                        href="https://easy-pv.co.uk/speedy-pv/demo"
                        className="project-link"
                        target="_blank"
                        rel="noreferrer"
                    >
                        View Live Demo
                    </a>
                </div>
            </section>
        </div>
    );
}
