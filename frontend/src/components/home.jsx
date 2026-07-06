import './pagestyles.css';
import './home.css'
import './font_kanit.css'
import { Link } from "react-router-dom";
import { VerticalTimeline } from './VerticalTimeline/VerticalTimeline'
import { Hero } from './Hero'

export default function HomePage(props) {
    return (
        <div>
            <Hero />
            <div className="home-content">
                <section className='section'>
                    <div className='card'>
                        <p className="intro">
                            My name is Reuben.
                        </p>
                        <p className="intro">
                            I am a web developer, and I like crafting things that I think are cool.
                        </p>
                        <p className="intro">
                            One of my recent projects has been&nbsp;
                            <a href="https://easy-pv.co.uk/speedy-pv/demo" target="_blank" rel="noreferrer">Speedy PV</a>,
                            a solar PV lead generator I've been working on at Midsummer Energy.
                        </p>
                        <p className="intro">
                            A previous project was building a <Link to="/machinelearner">general machine learner</Link>,
                            which is deployed on this site.
                        </p>
                        <p className="intro">
                            I've also built a <Link to="/dotnet-demo">C# .NET microservice backend</Link> to demonstrate
                            my ability to work with multiple tech stacks and microservices architecture.
                        </p>
                        <p className="intro">
                            I am also working on a Next.js game app, where players answer questions about movies, which I hope to
                            release soon.
                        </p>
                        <p className="intro">
                            Thanks for stopping by. If you have any comments or would like to get in touch,
                            you can reach me at reubenowenwilliams@outlook.com.
                        </p>
                    </div>
                </section>
                <section className='section'>
                    <div className='card'>
                        <h2>Application stack</h2>
                        <p className="intro">
                            The front end of this application is comprised of React, JSX, and CSS.
                            The back end uses a microservices architecture with a Flask/Python API for machine learning,
                            and a C# .NET microservice for additional functionality. This app is containerized
                            using Docker Compose for orchestration, and deployed using ECS with Fargate on AWS.
                        </p>
                    </div>
                </section>
                <section className='section'>
                    <div className='card'>
                        <h2>Site timeline</h2>
                        <VerticalTimeline />
                    </div>
                </section>
            </div>
        </div>
    )
}
