import './pagestyles.css';
import './home.css';
import { Link } from 'react-router-dom';

export default function ProjectsPage() {
    const projects = [
        {
            title: 'Machine Learner',
            description: 'A general-purpose machine learning platform that supports multiple model types including decision trees, k-nearest neighbours, and regression models. Features data preprocessing, scaling, and visualization tools. Deployed and running on this site.',
            link: '/machinelearner',
            label: 'Try it out'
        },

        {
            title: 'Speedy PV',
            description: 'A solar PV lead generator built while working at Midsummer Energy. Helps generate quotes and leads for solar panel installations.',
            link: 'https://easy-pv.co.uk/speedy-pv/demo',
            external: true,
            label: 'View demo'
        },
        {
            title: 'Movie Quiz Game',
            description: 'A Next.js multiplayer quiz game where players answer questions about movies. Currently in development and scheduled for release soon.',
            link: null,
            label: 'Coming soon',
            comingSoon: true
        }
    ];

    return (
        <div className="column-container">
            <section className="section">
                <div className="card">
                    <h2>Projects</h2>
                    <p className="intro">
                        Here's a summary of the projects I've worked on. Interactive demos are available below, along with descriptions of each project.
                    </p>
                </div>
            </section>
            <section className="section">
                <div className="card">
                    <div className="projects-grid">
                        {projects.map((project, index) => (
                            <div key={index} className="project-card">
                                <h3>{project.title}</h3>
                                <p>{project.description}</p>
                                {!project.comingSoon && project.link && (
                                    <a
                                        href={project.link}
                                        className="project-link"
                                        target={project.external ? "_blank" : undefined}
                                        rel={project.external ? "noreferrer" : undefined}
                                    >
                                        {project.label}
                                    </a>
                                )}
                                {project.comingSoon && (
                                    <span className="project-link coming-soon">{project.label}</span>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            </section>
        </div>
    );
}
