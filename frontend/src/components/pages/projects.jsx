import '../global.css';
import '../headernavbar.css';
import '../hero.css';
import '../home.css';
import '../font_kanit.css';
import { Link } from 'react-router-dom';
import { ProjectHero } from '../common/ProjectHero';

export default function ProjectsPage() {
    const projects = [
        {
            title: '2D Physics Simulation',
            description: 'An interactive physics sandbox with real-time ball-surface collision detection, gravity, and restitution. Draw surfaces, paint ball spawn areas, and watch thousands of balls bounce in parallel. Built with React, SignalR, and ASP.NET Core.',
            link: '/physics-simulation',
            label: 'Try It Out',
            image: '/assets/projects/2d_physics_comp.webm',
            imageColor: null,
            imageIcon: null
        },

        {
            title: 'Machine Learner',
            description: 'A general-purpose machine learning platform that supports multiple model types including decision trees, k-nearest neighbours, and regression models. Features data preprocessing, scaling, and visualization tools. Deployed and running on this site.',
            link: '/machinelearner',
            label: 'Try It Out',
            image: '/assets/projects/ml.webp',
            imageColor: null,
            imageIcon: null,
            specialClass: 'ml-card'
        },

        {
            title: 'Speedy PV',
            description: 'A solar PV lead generator built while working at Midsummer Energy. Helps generate quotes and leads for solar panel installations.',
            link: '/speedy-pv',
            label: 'Learn More',
            externalLink: 'https://easy-pv.co.uk/speedy-pv/demo',
            externalLabel: 'View Live Demo',
            image: '/assets/projects/speedypv_screenshot.webp',
            imageColor: null,
            imageIcon: null
        },

        {
            title: 'ML Vehicle Emissions',
            description: 'An investigation into UK vehicle CO2 emissions using machine learning. Analysed 6,756 vehicles with Python, MySQL, and XGBoost, tuning models via Bayesian optimisation.',
            link: '/ml-vehicle-emissions',
            label: 'Learn More',
            externalLink: 'https://github.com/ReubenGitHub/ML-Vehicle-Emissions',
            externalLabel: 'View on GitHub',
            image: '/assets/projects/car_exhaust.webp',
            imageColor: null,
            imageIcon: null
        },
        {
            title: 'Movie Quiz Game',
            description: 'A Next.js multiplayer quiz game where players answer questions about movies. Currently in development and scheduled for release soon.',
            link: null,
            label: 'Coming Soon',
            comingSoon: true,
            image: '/assets/projects/film_quiz_game.webp',
            imageColor: null,
            imageIcon: null
        }
    ];

    return (
        <div>
            <ProjectHero />
            <div className="column-container">
            <section className="section">
                <div className="projects-grid">
                        {projects.map((project, index) => (
                            <div key={index} className={`project-card${project.specialClass ? ' ' + project.specialClass : ''}`}>
                                {project.image && (
                                    <div className="project-card-image">
                                        {project.image.endsWith('.webm') || project.image.endsWith('.mp4') || project.image.endsWith('.ogg') ? (
                                            <video src={project.image} alt={project.title} autoPlay loop muted playsInline />
                                        ) : (
                                            <img src={project.image} alt={project.title} loading="lazy" />
                                        )}
                                    </div>
                                )}
                                {project.imageColor && (
                                    <div className="project-card-image project-card-placeholder" style={{backgroundColor: project.imageColor}}>
                                        <span className="project-card-icon">{project.imageIcon}</span>
                                    </div>
                                )}
                                <div className="project-card-content">
                                    <h3>{project.title}</h3>
                                    <p>{project.description}</p>
                                    {!project.comingSoon && project.link && (
                                        <div className="project-links">
                                            <a
                                                href={project.link}
                                                className="project-link"
                                            >
                                                {project.label}
                                            </a>
                                            {project.externalLink && (
                                                <a
                                                    href={project.externalLink}
                                                    className="project-link external-link"
                                                    target="_blank"
                                                    rel="noreferrer"
                                                >
                                                    {project.externalLabel}
                                                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 -960 960 960" fill="currentColor" style={{verticalAlign: 'middle', marginLeft: 4}}><path d="M200-120q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h280v80H200v560h560v-280h80v280q0 33-23.5 56.5T760-120H200Zm188-212-56-56 372-372H560v-80h280v280h-80v-144L388-332Z"/></svg>
                                                </a>
                                            )}
                                        </div>
                                    )}
                                    {project.comingSoon && (
                                        <span className="project-link coming-soon">{project.label}</span>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
            </section>
            </div>
        </div>
    );
}
