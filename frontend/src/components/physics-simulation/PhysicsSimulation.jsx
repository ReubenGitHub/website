import { useState, useEffect, useRef } from 'react';
import { HubConnectionBuilder, HttpTransportType, HubConnectionState } from '@microsoft/signalr';
import SurfaceDrawer from './SurfaceDrawer';
import SpawnControls from './SpawnControls';
import SimulationCanvas from './SimulationCanvas';
import SimulationControls from './SimulationControls';
import './PhysicsSimulation.css';

const PhysicsSimulation = () => {
  const [surface, setSurface] = useState([]);
  const ballsRef = useRef([]);
  const mountedRef = useRef(false);
  const [ballCount, setBallCount] = useState(2000);
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);
  const [activeBallCount, setActiveBallCount] = useState(0);
  const connectionRef = useRef(null);

  useEffect(() => {
    mountedRef.current = true;
    console.log('[PhysicsSim] Creating SignalR connection...');
    console.log('[PhysicsSim] Current location:', window.location.href);
    
    const connection = new HubConnectionBuilder()
      .withUrl('/physicsHub', {
        transport: HttpTransportType.WebSocket,
        accessTokenFactory: () => null,
      })
      .configureLogging('information')
      .build();

    connection.on('StateUpdate', (state) => {
      if (!mountedRef.current) return;
      if (state && state.balls && state.balls.length > 0) {
        console.log('[PhysicsSim] StateUpdate:', state.balls.length, 'balls, first:', state.balls[0]);
        ballsRef.current = state.balls;
        const active = state.balls.filter(b => b.active).length;
        setActiveBallCount(active);
      }
    });
    
    console.log('[PhysicsSim] Connection methods:', Object.keys(connection));

    connection.on('SimulationStarted', (state) => {
      console.log('[PhysicsSim] SimulationStarted received');
      if (mountedRef.current) {
        setIsRunning(true);
        setIsPaused(false);
        if (state?.balls && state.balls.length > 0) {
          ballsRef.current = state.balls;
          const active = state.balls.filter(b => b.active).length;
          setActiveBallCount(active);
        }
      }
    });

    connection.on('SimulationPaused', () => {
      console.log('[PhysicsSim] SimulationPaused received');
      if (mountedRef.current) {
        setIsRunning(false);
        setIsPaused(true);
      }
    });

    connection.on('SimulationResumed', () => {
      console.log('[PhysicsSim] SimulationResumed received');
      if (mountedRef.current) {
        setIsRunning(true);
        setIsPaused(false);
      }
    });

    connection.on('SimulationReset', (state) => {
      console.log('[PhysicsSim] SimulationReset received');
      if (mountedRef.current) {
        ballsRef.current = state?.balls || [];
        setActiveBallCount(0);
        setIsRunning(false);
        setIsPaused(false);
      }
    });

    connection.onclose((err) => {
      console.log('[PhysicsSim] Connection closed:', err);
      if (mountedRef.current) {
        setIsConnected(false);
        setIsRunning(false);
        setIsPaused(false);
        ballsRef.current = [];
        setActiveBallCount(0);
      }
    });

    connection.onreconnecting((err) => {
      console.log('[PhysicsSim] Reconnecting:', err);
      if (mountedRef.current) setIsConnected(false);
    });

    connection.onreconnected((connectionId) => {
      console.log('[PhysicsSim] Reconnected:', connectionId);
      if (mountedRef.current) setIsConnected(true);
    });

    connectionRef.current = connection;

    // Start connection with retry logic
    console.log('[PhysicsSim] Starting SignalR connection...');
    let attempts = 0;
    const maxAttempts = 10;
    
    const tryConnect = async () => {
      try {
        attempts++;
        await connection.start();
        if (!mountedRef.current) {
          console.log('[PhysicsSim] Component unmounted during connection, stopping...');
          await connection.stop();
          return;
        }
        console.log(`[PhysicsSim] SignalR connected successfully on attempt ${attempts}! ConnectionId:`, connection.connectionId);
        setIsConnected(true);
      } catch (err) {
        if (!mountedRef.current) return; // Component unmounted
        if (attempts >= maxAttempts) {
          console.error(`[PhysicsSim] SignalR connection failed after ${maxAttempts} attempts:`, err.message);
          if (mountedRef.current) setIsConnected(false);
          return;
        }
        const delay = Math.min(1000 * Math.pow(2, attempts), 10000);
        console.log(`[PhysicsSim] Connection attempt ${attempts}/${maxAttempts} failed, retrying in ${delay}ms...`);
        setTimeout(tryConnect, delay);
      }
    };
    
    tryConnect();

    return () => {
      console.log('[PhysicsSim] Cleaning up SignalR connection...');
      mountedRef.current = false;
      if (connectionRef.current) {
        connectionRef.current.stop();
      }
    };
  }, []);

  // Auto-select default surface on mount
  useEffect(() => {
    const defaultSurface = [
      { x: 350, y: 350 },
      { x: 600, y: 400 },
      { x: 850, y: 350 }
    ];
    setSurface(defaultSurface);
    console.log('[PhysicsSim] Auto-selected default surface');
  }, []);

  const handleSurfaceDrawn = (surfacePoints) => {
    console.log('[PhysicsSim] Surface drawn:', surfacePoints.length, 'points');
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

  const handlePlay = async () => {
    console.log('[PhysicsSim] handlePlay called, isPaused:', isPaused, 'ballCount:', ballCount);
    console.log('[PhysicsSim] Surface points:', surface.length);
    console.log('[PhysicsSim] IsConnected:', isConnected);
    setError(null);

    if (!surface || surface.length < 2) {
      console.error('[PhysicsSim] No surface drawn');
      setError('Please draw or select a surface first');
      return;
    }

    const isValid = await validateConfig(ballCount);
    if (!isValid) return;

    try {
      const connected = await ensureConnected();
      if (!connected) {
        setError('Connection lost. Please refresh the page.');
        return;
      }

      if (isPaused) {
        // Resume paused simulation (preserves ball positions)
        console.log('[PhysicsSim] Resuming simulation...');
        await connectionRef.current.invoke('ResumeSimulation', 
          { ballCount: ballCount, gravity: 9.8, restitution: 0.7, airResistance: 0.01, deltaTime: 1.0 / 30.0 },
          surface);
        console.log('[PhysicsSim] ResumeSimulation invoked successfully');
      } else {
        // Start fresh simulation
        console.log('[PhysicsSim] Starting new simulation...');
        await connectionRef.current.invoke('StartSimulation', 
          { ballCount: ballCount, gravity: 9.8, restitution: 0.7, airResistance: 0.01, deltaTime: 1.0 / 30.0 },
          surface);
        console.log('[PhysicsSim] StartSimulation invoked successfully');
      }
    } catch (err) {
      console.error('[PhysicsSim] Play error:', err);
      setError(`Failed to play simulation: ${err.message}`);
    }
  };

  const ensureConnected = async () => {
    if (!connectionRef.current) return false;
    
    if (connectionRef.current.state === 'Connected') {
      return true;
    }
    
    console.log('[PhysicsSim] Connection not connected (state:', connectionRef.current.state, '), reconnecting...');
    let reconnectAttempts = 0;
    let connected = false;
    while (reconnectAttempts < 5 && !connected) {
      try {
        reconnectAttempts++;
        await connectionRef.current.start();
        connected = true;
        setIsConnected(true);
        console.log('[PhysicsSim] Reconnected successfully on attempt', reconnectAttempts);
      } catch (e) {
        console.warn(`[PhysicsSim] Reconnect attempt ${reconnectAttempts} failed:`, e.message);
        await new Promise(r => setTimeout(r, 1000 * reconnectAttempts));
      }
    }
    return connected;
  };

  const handlePause = async () => {
    try {
      const connected = await ensureConnected();
      if (!connected) {
        setError('Connection lost. Please refresh the page.');
        return;
      }
      await connectionRef.current.invoke('PauseSimulation');
    } catch (err) {
      console.error('Pause error:', err);
      setError('Failed to pause: ' + err.message);
    }
  };



  const handleReset = async () => {
    try {
      const connected = await ensureConnected();
      if (!connected) {
        setError('Connection lost. Please refresh the page.');
        return;
      }
      await connectionRef.current.invoke('ResetSimulation');
      ballsRef.current = [];
      setActiveBallCount(0);
      setIsRunning(false);
    } catch (err) {
      console.error('Reset error:', err);
      setError('Failed to reset: ' + err.message);
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
            <h2>2. Set Ball Count</h2>
            <SpawnControls
              ballCount={ballCount}
              setBallCount={setBallCount}
            />
          </div>
        </div>

        <div className="simulation-right">
          <div className="panel canvas-panel">
            <div className="canvas-header">
              <h2>3. Simulation</h2>
              <SimulationControls
                isRunning={isRunning}
                isPaused={isPaused}
                onPlay={handlePlay}
                onPause={handlePause}
                onReset={handleReset}
              />
            </div>
            <div className="canvas-container">
              <div className="simulation-stats">
                <span>Active: {activeBallCount} balls</span>
              </div>
              <SimulationCanvas
                ballsRef={ballsRef}
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
