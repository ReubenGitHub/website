import './pagestyles.css';
import './home.css'
import './font_kanit.css'
import { Link } from "react-router-dom";
import { Hello } from './HelloWord/Hello'
import { VerticalTimeline } from './VerticalTimeline/VerticalTimeline'

export default function HomePage(props) {
    return (
        <div>
            <h1 class="hello-container">
                <Hello />
            </h1>
            <center>
                <div class='section'>
                    <p class="intro">
                        My name is Reuben.
                    </p>
                    <p class="intro">
                        I am a web developer, and I like crafting things that I think are cool.
                    </p>
                    <p class="intro">
                        One of my recent projects has been&nbsp;
                        <a href="https://easy-pv.co.uk/speedy-pv/demo" target="_blank" rel="noreferrer">Speedy PV</a>,
                        a solar PV lead generator I've been working on at Midsummer Energy.
                    </p>
                    <p class="intro">
                        A previous project was building a <Link to="/machinelearner">general machine learner</Link>,
                        which is deployed on this site.
                    </p>
                    <p class="intro">
                        I am also working on a Next.js game app, where players answer questions about movies, which I hope to
                        release soon.
                    </p>
                    <p class="intro">
                        Thanks for stopping by. If you have any comments or would like to get in touch,
                        you can reach me at reubenowenwilliams@outlook.com.
                    </p>
                </div>
                <div class='section'>
                    <h2>Application stack</h2>
                    <p class="intro">
                        The front end of this application is comprised of React, JSX, and CSS.
                        The back end is made up of Flask and Python. This app is containerized
                        using Docker, and deployed using ECS with Fargate on AWS.
                    </p>
                </div>
                <div class='section'>
                    <h2>Site timeline</h2>
                    <VerticalTimeline />
                </div>
            </center>
        </div>
    )
}
