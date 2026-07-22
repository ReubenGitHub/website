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
  const [isSurfaceDrawingEnabled, setIsSurfaceDrawingEnabled] = useState(false);
  
  // Ball spawn area painting state
  // null = default spawn area (rectangle), [] = cleared (no spawn area), array = custom painted
  const [spawnPixels, setSpawnPixels] = useState(null);
  const [isBallPaintingEnabled, setIsBallPaintingEnabled] = useState(false);
  
  // Simulation parameters
  const [restitution, setRestitution] = useState(1);

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
    { x: 400, y: 320 },
    { x: 600, y: 520 },
    { x: 800, y: 320 }
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

  // Current surface for rendering and simulation
  const currentSurface = (drawPoints.length > 0 && !isCleared) ? drawPoints : (isCleared ? [] : (surfaceType === 'flat' ? flatSurface : defaultSurface));

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
    console.log('[PhysicsSim] Spawn pixels:', spawnPixels === null ? 'default' : spawnPixels.length);
    console.log('[PhysicsSim] IsConnected:', isConnected);
    console.log('[PhysicsSim] Using surface with', currentSurface?.length, 'points, isCleared:', isCleared);
    setError(null);

    const isValid = await validateConfig(ballCount);
    if (!isValid) return;

    try {
      const connected = await ensureConnected();
      if (!connected) {
        setError('Connection lost. Please refresh the page.');
        return;
      }

      // Generate spawn pixels for backend (null = default rectangle, [] = cleared)
      let spawnPixelsToSend = spawnPixels;
      console.log('[PhysicsSim] Before processing: spawnPixels is', spawnPixels === null ? 'null' : `array with ${spawnPixels.length} pixels`, spawnPixels);
      
      if (spawnPixels === null) {
        // Generate default rectangle pixels (20% width/height, centered, 25% from top)
        const canvasWidth = 1200;
        const canvasHeight = 600;
        const rectWidth = canvasWidth * 0.2;
        const rectHeight = canvasHeight * 0.2;
        const rectX = (canvasWidth - rectWidth) / 2;
        const rectY = canvasHeight * 0.25;
        spawnPixelsToSend = [];
        const step = 4;
        for (let y = rectY; y < rectY + rectHeight; y += step) {
          for (let x = rectX; x < rectX + rectWidth; x += step) {
            spawnPixelsToSend.push({ x: Math.round(x), y: Math.round(y) });
          }
        }
        console.log('[PhysicsSim] Default spawn area: generated', spawnPixelsToSend.length, 'pixels');
      } else if (spawnPixels.length === 0) {
        // Cleared - no spawn area, backend will use surface-based spawning
        spawnPixelsToSend = [];
      }

      console.log('[PhysicsSim] spawnPixelsToSend:', spawnPixelsToSend === null ? 'null' : `array with ${spawnPixelsToSend.length} pixels`, spawnPixelsToSend?.slice(0, 5));

      const config = { 
        ballCount: ballCount, 
        gravity: 4.0, 
        restitution, 
        deltaTime: 1.0 / 30.0,
        spawnPixels: spawnPixelsToSend
      };

      if (isPaused) {
        // Resume paused simulation (preserves ball positions)
        console.log('[PhysicsSim] Resuming simulation...');
        await connectionRef.current.invoke('ResumeSimulation', config, currentSurface);
        console.log('[PhysicsSim] ResumeSimulation invoked successfully');
      } else {
        // Start fresh simulation
        console.log('[PhysicsSim] Starting new simulation...');
        await connectionRef.current.invoke('StartSimulation', config, currentSurface);
        console.log('[PhysicsSim] StartSimulation invoked successfully');
      }
    } catch (err) {
      console.error('[PhysicsSim] Play error:', err);
      setError(`Failed to play simulation: ${err.message}`);
    }
  };

  const handleToggleBallPainting = () => {
    const newValue = !isBallPaintingEnabled;
    setIsBallPaintingEnabled(newValue);
    // Mutual exclusivity: disabling surface drawing when enabling ball painting
    if (newValue) {
      setIsSurfaceDrawingEnabled(false);
    }
    console.log('[PhysicsSim] Ball painting toggled:', newValue, 'Surface drawing:', !newValue ? !isBallPaintingEnabled : isSurfaceDrawingEnabled);
  };

  const handleDefaultSpawnArea = () => {
    // Generate default spawn area rectangle: 20% width/height, centered, 25% from top
    // Set spawnPixels to null so UnifiedCanvas renders default rectangle overlay
    setSpawnPixels(null);
    console.log('[PhysicsSim] Default spawn area activated');
  };

  const handleClearSpawnArea = () => {
    setSpawnPixels([]);
    console.log('[PhysicsSim] Spawn area cleared');
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
      // Reset surface drawing only (preserve ball spawn area)
      setIsSurfaceDrawingEnabled(false);
    } catch (err) {
      console.error('Reset error:', err);
      setError('Failed to reset: ' + err.message);
    }
  };

  const handleRestitutionChange = async (newRestitution) => {
    setRestitution(newRestitution);
    try {
      const connected = await ensureConnected();
      if (connected) {
        await connectionRef.current.invoke('UpdateRestitution', newRestitution);
      }
    } catch (err) {
      console.error('Restitution update error:', err);
    }
  };

  return (
    <div className="physics-simulation-page">
      <div className="simulation-header">
        <h1>2D Physics Simulation</h1>
        <p>Draw a surface to bounce balls on</p>
      </div>

      {error && (
        <div className="error-message">
          <span>{error}</span>
          <button onClick={() => setError(null)}>×</button>
        </div>
      )}

      <div className="simulation-card">
        {/* Configuration section - side by side sliders */}
        <div className="simulation-card-controls">
          <div className="controls-section">
            <SpawnControls
              ballCount={ballCount}
              setBallCount={setBallCount}
              restitution={restitution}
              onRestitutionChange={handleRestitutionChange}
              controlsDisabled={isRunning}
            />
          </div>
        </div>

        {/* Surface/Canvas toolbar - glassmorphism card */}
        <div className="canvas-toolbar">
          {/* Ball spawn group */}
          <span className="toolbar-label">Balls:</span>
          <button 
            onClick={handleDefaultSpawnArea} 
            className="toolbar-btn ball-btn"
            disabled={isRunning}
            title="Use Default Spawn Area"
          >
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round">
              <rect x="3" y="5" width="12" height="8" rx="1" fill="currentColor" stroke="none"/>
            </svg>
          </button>
          <button 
            onClick={handleClearSpawnArea} 
            className="toolbar-btn ball-btn toolbar-btn-clear"
            disabled={isRunning}
            title="Clear Ball Spawn Area"
          >
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
              <path d="M4 4L14 14M14 4L4 14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
          </button>
          <button 
            onClick={handleToggleBallPainting} 
            className={`toolbar-btn ball-btn paintbrush-btn ${isBallPaintingEnabled ? 'active' : ''}`}
            disabled={isRunning}
            title="Paint Ball Spawn Area"
          >
            <svg width="18" height="18" viewBox="0 0 117.41 103.78" fill="none" stroke="currentColor" strokeWidth="8" strokeLinejoin="round">
              <path d="M0,103.78c11.7-8.38,30.46.62,37.83-14a16.66,16.66,0,0,0,.62-13.37,10.9,10.9,0,0,0-3.17-4.35,11.88,11.88,0,0,0-2.11-1.35c-9.63-4.78-19.67,1.91-25,10-4.9,7.43-7,16.71-8.18,23.07ZM54.09,43.42a54.31,54.31,0,0,1,15,18.06l50.19-49.16c3.17-3,5-5.53,2.3-10.13A6.5,6.5,0,0,0,117.41,0,7.09,7.09,0,0,0,112.8,1.6L54.09,43.42Zm-16.85,22c2.82,1.52,6.69,5.25,7.61,9.32L65.83,64c-3.78-7.54-8.61-14-15.23-18.58-6.9,9.27-5.5,11.17-13.36,20Z"/>
            </svg>
          </button>
          
          <label className="brush-label-inline">
            <span>Brush</span>
            <input
              type="range"
              min="4"
              max="20"
              value={brushSize}
              onChange={(e) => setBrushSize(Number(e.target.value))}
              className="toolbar-brush-slider"
              disabled={isRunning}
            />
            <span className="brush-value">{brushSize}px</span>
          </label>
          
          <div className="toolbar-divider"></div>
          
          {/* Surface drawing group */}
          <span className="toolbar-label">Surface:</span>
          <button 
            onClick={handleUseDefaultSurface} 
            className="toolbar-btn surface-btn-v"
            disabled={isRunning}
            title="V-Shape Surface"
          >
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path d="M3 10 L10 17 L17 10" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
          <button 
            onClick={handleUseFlatSurface} 
            className="toolbar-btn surface-btn-flat"
            disabled={isRunning}
            title="Flat Surface"
          >
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <line x1="3" y1="10" x2="17" y2="10" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </button>
          <button 
            onClick={handleClearDrawing} 
            className="toolbar-btn toolbar-btn-clear"
            disabled={isRunning}
            title="Clear Drawing"
          >
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
              <path d="M4 4L14 14M14 4L4 14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
          </button>
          <button 
            onClick={() => {
              const newValue = !isSurfaceDrawingEnabled;
              setIsSurfaceDrawingEnabled(newValue);
              if (newValue) setIsBallPaintingEnabled(false);
            }}
            className={`toolbar-btn pencil-btn ${isSurfaceDrawingEnabled ? 'active' : ''}`}
            disabled={isRunning}
            title="Enable Surface Drawing"
          >
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M11.5 2.5l4 4L6 16H2v-4L11.5 2.5z"/>
              <path d="M10 4l4 4"/>
            </svg>
          </button>
        </div>

        {/* Canvas section */}
        <div className="simulation-card-canvas">
          <div className={`canvas-with-controls ${isSurfaceDrawingEnabled ? 'drawing-mode' : ''} ${isBallPaintingEnabled ? 'paint-mode' : ''}`}>
            <UnifiedCanvas
              ballsRef={ballsRef}
              surface={currentSurface}
              isRunning={isRunning}
              isPaused={isPaused}
              drawPoints={drawPoints}
              setDrawPoints={setDrawPoints}
              brushSize={brushSize}
              isCleared={isCleared}
              isDrawing={isDrawing}
              setIsDrawing={setIsDrawing}
              surfaceType={surfaceType}
              isSurfaceDrawingEnabled={isSurfaceDrawingEnabled}
              isBallPaintingEnabled={isBallPaintingEnabled}
              spawnPixels={spawnPixels}
              setSpawnPixels={setSpawnPixels}
              defaultSurface={defaultSurface}
              flatSurface={flatSurface}
            />
            <SimulationControls
              isRunning={isRunning}
              isPaused={isPaused}
              onPlay={handlePlay}
              onPause={handlePause}
              onReset={handleReset}
            />
            <div className="simulation-stats">
              <span>Active: {activeBallCount} balls</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PhysicsSimulation;
