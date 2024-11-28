import React, {useState, useEffect} from 'react';
import {BrowserRouter as Router, Route, Routes} from "react-router-dom";
import HomePage from './components/home';
import {MLerPage} from './components/machinelearner';
import PFerPage from './components/pathfinder';
import {NavBar} from './components/headernavbar';

function App() {
  const [navState, setNavState] = useState(false);

  const Home = <HomePage />;
  const MLer = <MLerPage />;
  const PFer = <PFerPage />;

  const callbackFunctionHome = () => {
    setNavState(!navState);
  }   
  
  return (
    <Router>
      <NavBar navState={navState} />
      <Routes>
        <Route path="/" element = {Home} parentCallback = {callbackFunctionHome} />
        <Route path="/home" element = {Home} parentCallback = {callbackFunctionHome} />
        <Route path="/machinelearner" element = {MLer} />
        <Route path="/pathfinder" element = {PFer} />
      </Routes>
    </Router>
  );
}

export default App;