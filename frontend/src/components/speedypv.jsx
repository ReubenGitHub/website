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
                        className="project-link external-link"
                        target="_blank"
                        rel="noreferrer"
                    >
                        View Live Demo
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 -960 960 960" fill="currentColor" style={{verticalAlign: 'middle', marginLeft: 4}}><path d="M200-120q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h280v80H200v560h560v-280h80v280q0 33-23.5 56.5T760-120H200Zm188-212-56-56 372-372H560v-80h280v280h-80v-144L388-332Z"/></svg>
                    </a>
                </div>
            </section>
        </div>
    );
}
