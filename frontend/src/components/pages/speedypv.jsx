import '../global.css';
import '../headernavbar.css';
import '../Hero.css';
import '../home.css';

export default function SpeedyPVPage() {
    return (
        <div className="column-container project-page-wide">
            <section className="section">
                <div className="card">
                    <h2>Speedy PV</h2>
                    <p className="intro">
                        While working at <a href="https://midsummerwholesale.co.uk/" target="_blank" rel="noreferrer" className="external-link">Midsummer Energy</a>, I built a solar PV leads generator
                        to help homeowners quickly and easily get quotes for PV systems on their homes.
                    </p>
                    <p className="intro">
                        The tool was embedded on PV installer websites through a simple HTML script tag.
                        Generated leads were loaded onto the installer's Easy PV account,
                        connecting installers with potential customers.
                    </p>
                    <p className="intro">
                        I built the frontend using JS, HTML, and CSS. The backend consisted of Node.js endpoints
                        on the main Easy PV API. The automatic roof-scanning was provided by a separate microservice.
                    </p>
                    <div style={{textAlign: 'center', marginBottom: 20}}>
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
                    <div className="project-videos">
                        <div className="video-caption-group">
                            <div className="video-container video-container-desktop">
                                <video autoPlay loop muted playsInline className="project-video desktop-video">
                                    <source src="/assets/projects/speedypv_desktop.webm" type="video/webm" />
                                    Your browser does not support the video tag.
                                </video>
                            </div>
                            <span className="video-caption">Desktop view</span>
                        </div>
                        <div className="video-caption-group">
                            <div className="video-container video-container-mobile">
                                <video autoPlay loop muted playsInline className="project-video">
                                    <source src="/assets/projects/speedypv_mobile.webm" type="video/webm" />
                                    Your browser does not support the video tag.
                                </video>
                            </div>
                            <span className="video-caption">Mobile view</span>
                        </div>
                    </div>
                </div>
            </section>
        </div>
    );
}
