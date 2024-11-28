import './pagestyles.css';
import {Link} from "react-router-dom";

function HomePage(props) {
  return (
    <html>
    <body>

      <div class="about-section">
          <h1>Hello!</h1>
          <center><p class="intro">
            Hi, my name is Reuben.
            <br></br>
            <br></br>
            I am a studying actuarial consultant, and I have a passion for crafting solutions to interesting technical problems.
            I was raised in Essex, I studied Mathematics at the University of Warwick, and I now reside in Cambridge.
            When I have time, I also like to play guitar, write music, and boulder.
            <br></br>
            <br></br>
            This website was created entirely by myself, both as an undertaking to learn new technologies, and to establish a platform to host my various other projects.
            I am keen to showcase my learning in a practical manner, and I am particularly interested in machine learning; hence, I created a&nbsp; 
            <Link to="/machinelearner" onClick={ () => props.parentCallback()}>
              general machine learner
            </Link>
            &nbsp;as my first app!<br></br>
            <br></br>
            Future plans for the site include enhancements to the model types and representations in the Machine Learner, and the deployment of my path-finding application.
            <br></br>
            <br></br>
            Thank you for stopping by, I hope you enjoy the experience. If you have any comments or would like to get in touch, you can reach me at reubenowenwilliams@outlook.com.
            <br></br>
            <br></br>
            <br></br>
            Reuben
            <br></br>
            <br></br>
            <br></br>
            <br></br>
            <br></br>
            <br></br>
            <br></br>
            <br></br>
            <br></br>
            <br></br>
            <br></br>
            <br></br>
            <br></br>
            <br></br>
            <br></br>
            <br></br>
            <br></br>
            <br></br>
            <i>
              This website was created using a <b>React</b> and <b>jsx</b> front-end, a <b>Flask</b> api, a Gunicorn server, and a <b>Python</b> back-end.
              Development of this site began on the 29th December 2021.
              On the 7th February 2022, I deployed the first fully-functioning version of the site on <b>Heroku</b>.
            </i>
          </p></center>
      </div>
    
    </body>
    </html>
  );
}

export default HomePage;