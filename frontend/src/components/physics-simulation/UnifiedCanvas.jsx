import { useRef, useEffect, useCallback } from 'react';
import './SimulationCanvas.css';

const UnifiedCanvas = ({
  ballsRef,
  surface,
  isRunning,
  isPaused,
  drawPoints,
  brushSize,
  isCleared,
  isDrawing,
  setIsDrawing,
  setDrawPoints,
  surfaceType,
  isSurfaceDrawingEnabled,
  defaultSurface,
  flatSurface
}) => {
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);
  const surfaceRef = useRef([]);
  const defaultSurfaceRef = useRef(defaultSurface || []);
  const flatSurfaceRef = useRef(flatSurface || []);
  const runningRef = useRef(false);
  const pausedRef = useRef(false);
  const drawPointsRef = useRef(drawPoints);
  const brushSizeRef = useRef(brushSize);
  const isClearedRef = useRef(isCleared);
  const isDrawingRef = useRef(isDrawing);
  const surfaceTypeRef = useRef(surfaceType);
  const isSurfaceDrawingEnabledRef = useRef(isSurfaceDrawingEnabled);
  const currentStrokeRef = useRef([]); // Batched points during active stroke
  const connectorEndRef = useRef(null); // End point of connector line (from old surface to new stroke start)

  useEffect(() => { surfaceRef.current = surface || []; }, [surface]);
  useEffect(() => { defaultSurfaceRef.current = defaultSurface || []; }, [defaultSurface]);
  useEffect(() => { flatSurfaceRef.current = flatSurface || []; }, [flatSurface]);
  useEffect(() => { runningRef.current = isRunning; }, [isRunning]);
  useEffect(() => { pausedRef.current = isPaused; }, [isPaused]);
  useEffect(() => { drawPointsRef.current = drawPoints; }, [drawPoints]);
  useEffect(() => { brushSizeRef.current = brushSize; }, [brushSize]);
  useEffect(() => { isClearedRef.current = isCleared; }, [isCleared]);
  useEffect(() => { isDrawingRef.current = isDrawing; }, [isDrawing]);
  useEffect(() => { surfaceTypeRef.current = surfaceType; }, [surfaceType]);
  useEffect(() => { isSurfaceDrawingEnabledRef.current = isSurfaceDrawingEnabled; }, [isSurfaceDrawingEnabled]);

  useEffect(() => {
    const handler = (e) => {
      const pos = e.detail;
      console.log('[UnifiedCanvas] Draw point event received:', pos, 'current points:', drawPointsRef.current.length);
      if (pos && (!drawPointsRef.current.length || 
          Math.sqrt((pos.x - drawPointsRef.current[drawPointsRef.current.length-1].x) ** 2 + 
                    (pos.y - drawPointsRef.current[drawPointsRef.current.length-1].y) ** 2) >= 5)) {
        setDrawPoints(prev => {
          const newPoints = [...prev, pos];
          console.log('[UnifiedCanvas] setDrawPoints called, new length:', newPoints.length);
          return newPoints;
        });
      } else {
        console.log('[UnifiedCanvas] Point rejected (too close to last point)');
      }
    };
    window.addEventListener('unifiedCanvasDrawPoint', handler);
    return () => window.removeEventListener('unifiedCanvasDrawPoint', handler);
  }, [setDrawPoints]);

  const getCanvasPosition = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
     const scaleX = canvas.width / rect.width;
     const scaleY = canvas.height / rect.height;
    return {
       x: (e.clientX - rect.left) * scaleX,
       y: (e.clientY - rect.top) * scaleY
    };
  }, []);

  const drawGrid = useCallback((ctx, width, height) => {
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 50) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 50) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
  }, []);

  const drawDefaultSurface = useCallback((ctx) => {
    const pts = defaultSurfaceRef.current;
    if (pts.length < 2) return;
    ctx.strokeStyle = 'rgba(100, 200, 255, 0.8)';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
    ctx.stroke();
  }, []);

  const drawFlatSurface = useCallback((ctx, height) => {
    const pts = flatSurfaceRef.current;
    if (pts.length < 2) return;
    ctx.strokeStyle = 'rgba(100, 200, 255, 0.8)';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
    ctx.stroke();
  }, []);

  const drawStroke = useCallback((ctx, points) => {
    if (points.length < 2) return;
    ctx.strokeStyle = 'rgba(100, 200, 255, 0.8)';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
    ctx.stroke();
  }, []);

  const drawCustomSurface = useCallback((ctx, points) => {
    if (points.length < 2) return;
    ctx.strokeStyle = 'rgba(100, 200, 255, 0.8)';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.stroke();
  }, []);

  const drawSimulationSurface = useCallback((ctx, surface) => {
    if (surface.length < 2) return;
    ctx.strokeStyle = 'rgba(100, 200, 255, 0.9)';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(surface[0].x, surface[0].y);
    for (let i = 1; i < surface.length; i++) {
      ctx.lineTo(surface[i].x, surface[i].y);
    }
    ctx.stroke();
  }, []);

  const drawBalls = useCallback((ctx, balls) => {
    const activeBalls = balls.filter(b => b.active);
    if (activeBalls.length > 0) {
      const ballsByColor = new Map();
      activeBalls.forEach(ball => {
        const key = `${ball.r},${ball.g},${ball.b}`;
        if (!ballsByColor.has(key)) ballsByColor.set(key, []);
        ballsByColor.get(key).push(ball);
      });
      ballsByColor.forEach((ballGroup, key) => {
        const [r, g, b] = key.split(',').map(Number);
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.shadowColor = `rgba(${r}, ${g}, ${b}, 0.5)`;
        ctx.shadowBlur = 3;
        ballGroup.forEach(ball => {
          ctx.beginPath();
          ctx.arc(ball.x, ball.y, ball.radius, 0, Math.PI * 2);
          ctx.fill();
        });
      });
      ctx.shadowBlur = 0;
    }
  }, []);

  const render = useCallback((dp) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    drawGrid(ctx, width, height);

    // In drawing mode (not running, not paused): show default or custom surface overlay
    if (!runningRef.current && !pausedRef.current) {
      const currentDp = dp !== undefined ? dp : drawPointsRef.current;
      console.log('[Canvas] render(drawing mode) - drawPoints:', currentDp.length, 'isDrawing:', isDrawingRef.current, 'cleared:', isClearedRef.current, 'surfaceType:', surfaceTypeRef.current);
      // Always draw previous completed drawing first (if any)
      if (currentDp.length > 1 && !isDrawingRef.current) {
        console.log('[Canvas] render: drawing custom surface');
        drawCustomSurface(ctx, currentDp);
      } else if (currentDp.length > 1 && isDrawingRef.current) {
        console.log('[Canvas] render: drawing custom surface (while drawing)');
        drawCustomSurface(ctx, currentDp);
      } else if (!isClearedRef.current && !isDrawingRef.current) {
        // Show default surface only when not actively drawing and no completed points
        console.log('[Canvas] render: drawing default surface');
        if (surfaceTypeRef.current === 'flat') {
          drawFlatSurface(ctx, width, height);
        } else {
          drawDefaultSurface(ctx, width, height);
        }
      }
      // Then draw active stroke and connector on top (if actively drawing)
      if (isDrawingRef.current && currentStrokeRef.current.length > 0) {
        console.log('[Canvas] render: redrawing active stroke, points:', currentStrokeRef.current.length);
        drawStroke(ctx, currentStrokeRef.current);
        // Draw connector line from last completed point to stroke start (independent of stroke length)
        if (connectorEndRef.current && currentDp.length > 0) {
          const lastPoint = currentDp[currentDp.length - 1];
          console.log('[Canvas] render: DRAWING CONNECTOR from', lastPoint, 'to', connectorEndRef.current);
          ctx.strokeStyle = 'rgba(100, 200, 255, 0.8)';
          ctx.lineWidth = 3;
          ctx.lineCap = 'round';
          ctx.beginPath();
          ctx.moveTo(lastPoint.x, lastPoint.y);
          ctx.lineTo(connectorEndRef.current.x, connectorEndRef.current.y);
          ctx.stroke();
          console.log('[Canvas] render: CONNECTOR stroke() called');
        } else {
          console.log('[Canvas] render: NOT drawing connector - connectorEndRef:', connectorEndRef.current, 'currentDp.length:', currentDp.length);
        }
      }
    } else {
      // In simulation mode (running or paused): draw surface and balls
      // Draw custom surface if drawPoints exist, otherwise draw from surface ref
      if (drawPointsRef.current.length > 1) {
        drawSimulationSurface(ctx, drawPointsRef.current);
      } else if (!isClearedRef.current && surfaceRef.current.length >= 2) {
        drawSimulationSurface(ctx, surfaceRef.current);
      }
      if ((runningRef.current || pausedRef.current) && ballsRef.current && ballsRef.current.length > 0) {
        drawBalls(ctx, ballsRef.current);
      }
    }
    
    if (runningRef.current) {
      animationFrameRef.current = requestAnimationFrame(render);
    }
  }, [drawGrid, drawDefaultSurface, drawFlatSurface, drawCustomSurface, drawStroke, drawSimulationSurface, drawBalls]);

  // Render when drawing state changes
  useEffect(() => {
    console.log('[Canvas] useEffect[drawPoints] firing - drawPoints:', drawPoints.length, 'isDrawingRef:', isDrawingRef.current, 'strokeLen:', currentStrokeRef.current.length);
    render(drawPoints);
  }, [drawPoints, brushSize, isCleared, surfaceType, render]);


  // Start/stop animation loop when running state changes
  useEffect(() => {
    console.log('[Canvas] useEffect[isRunning] firing - isRunning:', isRunning, 'isPaused:', isPaused);
    if (runningRef.current) {
      animationFrameRef.current = requestAnimationFrame(() => render(drawPointsRef.current));
    } else {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      // Render once to show surface when stopped/paused
      console.log('[Canvas] useEffect[isRunning]: calling render() no args');
      render();
    }
  }, [isRunning, isPaused, render]);

  // Setup canvas and animation loop
  useEffect(() => {
    console.log('[Canvas] useEffect[canvas setup] firing - running:', runningRef.current);
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (runningRef.current) {
      animationFrameRef.current = requestAnimationFrame(render);
    }
    return () => {
      console.log('[Canvas] useEffect[canvas setup] cleanup');
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, [render]);

  const handleMouseDown = useCallback((e) => {
    console.log('[Canvas] mousedown - isRunning:', runningRef.current, 'drawingEnabled:', isSurfaceDrawingEnabledRef.current, 'prevStrokeLen:', currentStrokeRef.current.length);
    const isRunning = runningRef.current;

    if (isRunning) { console.log('[Canvas] mousedown BLOCKED: isRunning'); return; }
    if (!isSurfaceDrawingEnabledRef.current) { console.log('[Canvas] mousedown BLOCKED: drawing not enabled'); return; }
    // Clear existing surface when starting a new drawing session
    const prevStroke = currentStrokeRef.current;
    if (prevStroke.length > 0) {
      console.log('[Canvas] mousedown: committing prev stroke, points:', prevStroke.length);
      setDrawPoints(prev => [...prev, ...prevStroke]);
      currentStrokeRef.current = [];
    }
    const pos = getCanvasPosition(e);
    if (drawPoints.length > 0) {
      // Existing surface from previous drawing - draw connector line to new start
      const lastPoint = drawPoints[drawPoints.length - 1];
      connectorEndRef.current = pos;
      console.log('[Canvas] mousedown: drawing connector from', lastPoint, 'to', pos);
    } else {
      connectorEndRef.current = null;
    }
    isDrawingRef.current = true; // Update synchronously for immediate render
    setIsDrawing(true);
    console.log('[Canvas] mousedown: isDrawingRef set to true');
    currentStrokeRef.current = [pos]; // Start new stroke - MUST be BEFORE render
    // Force immediate redraw so V-shape disappears right away
    render(drawPoints);
    // Draw initial point
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.strokeStyle = 'rgba(100, 200, 255, 0.8)';
      ctx.lineWidth = 3;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();
      ctx.moveTo(pos.x, pos.y);
      ctx.stroke();
    }
  }, [getCanvasPosition, setIsDrawing, setDrawPoints, drawPoints, render]);

  const handleMouseMove = useCallback((e) => {
    if (!isDrawingRef.current || runningRef.current) { console.log('[Canvas] mousemove BLOCKED - drawing:', isDrawingRef.current, 'running:', runningRef.current); return; }
    if (!isSurfaceDrawingEnabledRef.current) { console.log('[Canvas] mousemove BLOCKED: not enabled'); return; }
    const pos = getCanvasPosition(e);
    // Add to stroke batch
    const stroke = currentStrokeRef.current;
    console.log('[Canvas] mousemove - strokeLen:', stroke.length, 'pos:', pos);
    if (stroke.length > 0) {
      const prev = stroke[stroke.length - 1];
      const dist = Math.sqrt((pos.x - prev.x) ** 2 + (pos.y - prev.y) ** 2);
      if (dist >= 5) {
        stroke.push(pos);
        console.log('[Canvas] mousemove: added point, total stroke:', stroke.length);
        // Draw directly to canvas (no React re-render)
        const canvas = canvasRef.current;
        if (canvas) {
          const ctx = canvas.getContext('2d');
          ctx.strokeStyle = 'rgba(100, 200, 255, 0.8)';
          ctx.lineWidth = 3;
          ctx.lineCap = 'round';
          ctx.lineJoin = 'round';
          ctx.beginPath();
          ctx.moveTo(prev.x, prev.y);
          ctx.lineTo(pos.x, pos.y);
          ctx.stroke();
        }
      } else {
        console.log('[Canvas] mousemove: point too close, skipped');
      }
    }
  }, [getCanvasPosition]);

  const handleMouseUp = useCallback(() => {
    const stroke = currentStrokeRef.current;
    console.log('[Canvas] mouseup - strokeLen:', stroke.length, 'drawPoints before:', drawPointsRef.current.length);
    if (stroke.length > 0) {
      // Commit batched points to React state (single update)
      setDrawPoints(prev => [...prev, ...stroke]);
      console.log('[Canvas] mouseup: committed stroke to drawPoints');
    }
    currentStrokeRef.current = [];
    isDrawingRef.current = false; // Update synchronously for immediate render
    setIsDrawing(false);
    console.log('[Canvas] mouseup: done, drawPoints after:', drawPointsRef.current.length);
  }, [setDrawPoints, setIsDrawing]);

  return (
    <canvas
      ref={canvasRef}
      width={1200}
      height={600}
      className="simulation-canvas"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={() => setIsDrawing(false)}
    />
  );
};

export default UnifiedCanvas;
