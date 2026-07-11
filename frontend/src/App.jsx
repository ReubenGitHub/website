import React, {useState, useEffect} from 'react';
import {BrowserRouter as Router, Route, Routes} from "react-router-dom";
import HomePage from './components/home';
import {MLerPage} from './components/machinelearner';
import {DotNetDemo} from './components/DotNetDemo';
import ProjectsPage from './components/projects';
import {NavBar} from './components/headernavbar';

function App() {
  const [navState, setNavState] = useState(false);

  const Home = <HomePage />;
  const MLer = <MLerPage />;
  const DotNet = <DotNetDemo />;
  const Projects = <ProjectsPage />;
  const callbackFunctionHome = () => {
    setNavState(!navState);
  }   
  
  return (
    <Router>
      <NavBar navState={navState} />
      {/* Global floating shapes background */}
      <div className="global-bg-shapes">
        <div className="global-shape global-shape-1" />
        <div className="global-shape global-shape-2" />
        <div className="global-shape global-shape-3" />
        <div className="global-shape global-shape-4" />
        <div className="global-shape global-shape-5" />
      </div>
      <Routes>
        <Route path="/" element = {Home} parentCallback = {callbackFunctionHome} />
        <Route path="/home" element = {Home} parentCallback = {callbackFunctionHome} />
        <Route path="/machinelearner" element = {MLer} />
        <Route path="/dotnet-demo" element = {DotNet} />
        <Route path="/projects" element = {Projects} />
      </Routes>
    </Router>
  );
}

export default App;