import { useState, useEffect, useRef } from 'react';
import { HubConnectionBuilder, HttpTransportType, HubConnectionState } from '@microsoft/signalr';
import SpawnControls from './SpawnControls';
import SimulationControls from './SimulationControls';
import UnifiedCanvas from './UnifiedCanvas';
import './PhysicsSimulation.css';

const PhysicsSimulation = () => {
  const ballsRef = useRef([]);
  const mountedRef = useRef(false);
  const [ballCount, setBallCount] = useState(2000);
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);
  const [activeBallCount, setActiveBallCount] = useState(0);
  const connectionRef = useRef(null);
  
  // Drawing state
  const [drawPoints, setDrawPoints] = useState([]);
  const [brushSize, setBrushSize] = useState(8);
  const [isDrawing, setIsDrawing] = useState(false);
  const [isCleared, setIsCleared] = useState(false);
  const [surfaceType, setSurfaceType] = useState('v'); // 'v' or 'flat'
  
  // Simulation parameters
  const [restitution, setRestitution] = useState(0.4);

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

  // Centered V-shape surface (1/4 off bottom, centered horizontally)
  const defaultSurface = [
    { x: 300, y: 350 },
    { x: 600, y: 450 },
    { x: 900, y: 350 }
  ];

  // Flat surface (centered horizontally, 1/4 off bottom)
  const flatSurface = [
    { x: 100, y: 450 },
    { x: 1100, y: 450 }
  ];

  const handleClearDrawing = () => {
    console.log('[PhysicsSim] Clearing drawing completely');
    setDrawPoints([]);
    setIsCleared(true);
  };

  const handleUseDefaultSurface = () => {
    console.log('[PhysicsSim] Using default V-surface');
    setDrawPoints([]);
    setIsCleared(false);
    setSurfaceType('v');
  };

  const handleUseFlatSurface = () => {
    console.log('[PhysicsSim] Using flat surface');
    setDrawPoints([]);
    setIsCleared(false);
    setSurfaceType('flat');
  };

  // Reset cleared state when user starts drawing
  useEffect(() => {
    if (drawPoints.length > 0) {
      setIsCleared(false);
    }
  }, [drawPoints]);

  const validateConfig = async (count) => {
    try {
      const response = await fetch('/api/simulation/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ballCount: count, gravity: 4.0, restitution })
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
    console.log('[PhysicsSim] Draw points:', drawPoints.length);
    console.log('[PhysicsSim] IsConnected:', isConnected);
    setError(null);

    // Use drawPoints if available and not cleared, otherwise use default surface
    const currentSurface = (drawPoints.length > 0 && !isCleared) ? drawPoints : (isCleared ? [] : (surfaceType === 'flat' ? flatSurface : defaultSurface));
    console.log('[PhysicsSim] Using surface with', currentSurface?.length, 'points, isCleared:', isCleared);

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
          { ballCount: ballCount, gravity: 4.0, restitution, deltaTime: 1.0 / 30.0 },
          currentSurface);
        console.log('[PhysicsSim] ResumeSimulation invoked successfully');
      } else {
        // Start fresh simulation
        console.log('[PhysicsSim] Starting new simulation...');
        await connectionRef.current.invoke('StartSimulation', 
          { ballCount: ballCount, gravity: 4.0, restitution, deltaTime: 1.0 / 30.0 },
          currentSurface);
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
          <div className="panel spawn-panel">
            <h2>Set Ball Count</h2>
            <SpawnControls
              ballCount={ballCount}
              setBallCount={setBallCount}
              restitution={restitution}
              setRestitution={setRestitution}
            />
          </div>
        </div>

        <div className="simulation-right">
          <div className="panel canvas-panel">
            <div className="canvas-header">
              <h2>Simulation</h2>
              <SimulationControls
                isRunning={isRunning}
                isPaused={isPaused}
                onPlay={handlePlay}
                onPause={handlePause}
                onReset={handleReset}
              />
            </div>
            
            {/* Canvas Controls */}
            <div className="canvas-controls">
              <span className="canvas-controls-label">Canvas Controls:</span>
              
              <button 
                onClick={handleUseDefaultSurface} 
                className="canvas-control-btn"
                disabled={isRunning || isPaused}
              >
                Use Default Surface
              </button>
              
              <button 
                onClick={handleUseFlatSurface} 
                className="canvas-control-btn"
                disabled={isRunning || isPaused}
              >
                Flat Surface
              </button>
              
              <button 
                onClick={handleClearDrawing} 
                className="canvas-control-btn btn-clear"
                disabled={isRunning || isPaused}
              >
                Clear Drawing
              </button>
              
              <label className="canvas-control-label">
                Brush: {brushSize}px
                <input
                  type="range"
                  min="4"
                  max="20"
                  value={brushSize}
                  onChange={(e) => setBrushSize(Number(e.target.value))}
                  className="canvas-control-slider"
                  disabled={isRunning || isPaused}
                />
              </label>
            </div>

            <div className="canvas-container">
              <div className="simulation-stats">
                <span>Active: {activeBallCount} balls</span>
              </div>
              <UnifiedCanvas
                ballsRef={ballsRef}
                isRunning={isRunning}
                isPaused={isPaused}
                drawPoints={drawPoints}
                setDrawPoints={setDrawPoints}
                brushSize={brushSize}
                isCleared={isCleared}
                isDrawing={isDrawing}
                setIsDrawing={setIsDrawing}
                surfaceType={surfaceType}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PhysicsSimulation;
