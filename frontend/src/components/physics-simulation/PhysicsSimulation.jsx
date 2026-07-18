import { useState, useEffect, useRef } from 'react';
import { HubConnectionBuilder, HttpTransportType } from '@microsoft/signalr';
import SurfaceDrawer from './SurfaceDrawer';
import SpawnControls from './SpawnControls';
import SimulationCanvas from './SimulationCanvas';
import SimulationControls from './SimulationControls';
import './PhysicsSimulation.css';

const PhysicsSimulation = () => {
  const [surface, setSurface] = useState([]);
  const [balls, setBalls] = useState([]);
  const [ballCount, setBallCount] = useState(2000);
  const [isRunning, setIsRunning] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);
  const connectionRef = useRef(null);

  useEffect(() => {
    // SignalR connection — use relative URL, Vite proxies to port 5001
    const connection = new HubConnectionBuilder()
      .withUrl('/physicsHub')
      .withAutomaticReconnect()
      .build();

    connection.on('StateUpdate', (state) => {
      if (state && state.balls) {
        setBalls(state.balls);
      }
    });

    connection.on('SimulationPaused', () => {
      setIsRunning(false);
    });

    connection.on('SimulationResumed', () => {
      setIsRunning(true);
    });

    connection.on('SimulationReset', (state) => {
      setBalls(state?.balls || []);
      setIsRunning(false);
    });

    connection.onclose(() => {
      setIsConnected(false);
      setIsRunning(false);
    });

    connectionRef.current = connection;

    // Catch connection errors via the start promise
    connection.start().then(() => {
      setIsConnected(true);
    }).catch((err) => {
      console.error('SignalR connection failed:', err);
      setIsConnected(false);
    });

    return () => {
      if (connectionRef.current) {
        connectionRef.current.stop();
      }
    };
  }, []);

  const handleSurfaceDrawn = (surfacePoints) => {
    setSurface(surfacePoints);
  };

  const validateConfig = async (count) => {
    try {
      const response = await fetch('/api/simulation/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ballCount: count, gravity: 9.8, restitution: 0.7 })
      });
      const result = await response.json();
      if (!result.valid) {
        setError(result.error || 'Invalid configuration');
        return false;
      }
      return true;
    } catch (err) {
      setError('Failed to validate configuration');
      return false;
    }
  };

  const handleStart = async (count) => {
    setError(null);

    if (!surface || surface.length < 2) {
      setError('Please draw or select a surface first');
      return;
    }

    const isValid = await validateConfig(count);
    if (!isValid) return;

    try {
      if (!isConnected && connectionRef.current) {
        await connectionRef.current.start();
        setIsConnected(true);
      }

      await connectionRef.current.invoke('StartSimulation', {
        ballCount: count,
        gravity: 9.8,
        restitution: 0.7,
        airResistance: 0.01,
        deltaTime: 1.0 / 30.0
      }, surface);

      setIsRunning(true);
    } catch (err) {
      setError(`Failed to start simulation: ${err.message}`);
      console.error('Start error:', err);
    }
  };

  const handlePause = async () => {
    try {
      await connectionRef.current?.invoke('PauseSimulation');
    } catch (err) {
      console.error('Pause error:', err);
    }
  };

  const handleResume = async () => {
    try {
      await connectionRef.current?.invoke('ResumeSimulation');
    } catch (err) {
      console.error('Resume error:', err);
    }
  };

  const handleReset = async () => {
    try {
      await connectionRef.current?.invoke('ResetSimulation');
      setBalls([]);
      setIsRunning(false);
    } catch (err) {
      console.error('Reset error:', err);
    }
  };

  return (
    <div className="physics-simulation-page">
      <div className="simulation-header">
        <h1>2D Physics Simulation</h1>
        <p>Draw a surface, set the ball count, and watch the simulation come to life!</p>
      </div>

      {error && (
        <div className="error-message">
          <span>{error}</span>
          <button onClick={() => setError(null)}>×</button>
        </div>
      )}

      <div className="simulation-content">
        <div className="simulation-left">
          <div className="panel surface-panel">
            <h2>1. Draw Your Surface</h2>
            <SurfaceDrawer onSurfaceDrawn={handleSurfaceDrawn} />
          </div>

          <div className="panel spawn-panel">
            <h2>2. Set Ball Count & Start</h2>
            <SpawnControls
              ballCount={ballCount}
              setBallCount={setBallCount}
              onStart={handleStart}
              onReset={handleReset}
              isRunning={isRunning}
            />
          </div>
        </div>

        <div className="simulation-right">
          <div className="panel canvas-panel">
            <div className="canvas-header">
              <h2>3. Simulation</h2>
              <SimulationControls
                isRunning={isRunning}
                onPause={handlePause}
                onResume={handleResume}
                onReset={handleReset}
              />
            </div>
            <div className="canvas-container">
              <SimulationCanvas
                balls={balls}
                surface={surface}
                isRunning={isRunning}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PhysicsSimulation;
