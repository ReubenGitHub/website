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
  surfaceType
}) => {
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);
  const surfaceRef = useRef([]);
  const runningRef = useRef(false);
  const pausedRef = useRef(false);
  const drawPointsRef = useRef(drawPoints);
  const brushSizeRef = useRef(brushSize);
  const isClearedRef = useRef(isCleared);
  const isDrawingRef = useRef(isDrawing);
  const surfaceTypeRef = useRef(surfaceType);

  useEffect(() => { surfaceRef.current = surface || []; }, [surface]);
  useEffect(() => { runningRef.current = isRunning; }, [isRunning]);
  useEffect(() => { pausedRef.current = isPaused; }, [isPaused]);
  useEffect(() => { drawPointsRef.current = drawPoints; }, [drawPoints]);
  useEffect(() => { brushSizeRef.current = brushSize; }, [brushSize]);
  useEffect(() => { isClearedRef.current = isCleared; }, [isCleared]);
  useEffect(() => { isDrawingRef.current = isDrawing; }, [isDrawing]);
  useEffect(() => { surfaceTypeRef.current = surfaceType; }, [surfaceType]);

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

  const drawDefaultSurface = useCallback((ctx, width, height) => {
    // Centered V-shape surface: (300,350), (600,450), (900,350)
    ctx.strokeStyle = 'rgba(100, 200, 255, 0.8)';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(300, 350);
    ctx.lineTo(600, 450);
    ctx.lineTo(900, 350);
    ctx.stroke();
    ctx.fillStyle = 'rgba(100, 200, 255, 0.05)';
    ctx.beginPath();
    ctx.moveTo(300, 350);
    ctx.lineTo(600, 450);
    ctx.lineTo(900, 350);
    ctx.closePath();
    ctx.fill();
  }, []);

  const drawFlatSurface = useCallback((ctx, width, height) => {
    // Flat surface: (100,450), (1100,450)
    ctx.strokeStyle = 'rgba(100, 200, 255, 0.8)';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(100, 450);
    ctx.lineTo(1100, 450);
    ctx.stroke();
    ctx.fillStyle = 'rgba(100, 200, 255, 0.05)';
    ctx.beginPath();
    ctx.moveTo(100, 450);
    ctx.lineTo(1100, 450);
    ctx.lineTo(1100, height);
    ctx.lineTo(100, height);
    ctx.closePath();
    ctx.fill();
  }, []);

  const drawCustomSurface = useCallback((ctx, points, size) => {
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
    ctx.fillStyle = 'rgba(100, 200, 255, 0.3)';
    points.forEach(point => {
      ctx.beginPath();
      ctx.arc(point.x, point.y, size / 2, 0, Math.PI * 2);
      ctx.fill();
    });
  }, []);

  const drawSimulationSurface = useCallback((ctx, surface) => {
    if (surface.length < 2) return;
    ctx.fillStyle = 'rgba(100, 200, 255, 0.08)';
    ctx.beginPath();
    ctx.moveTo(surface[0].x, surface[0].y);
    for (let i = 1; i < surface.length; i++) {
      ctx.lineTo(surface[i].x, surface[i].y);
    }
    ctx.closePath();
    ctx.fill();
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
    ctx.shadowColor = 'rgba(100, 200, 255, 0.5)';
    ctx.shadowBlur = 10;
    ctx.stroke();
    ctx.shadowBlur = 0;
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

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    drawGrid(ctx, width, height);
    
    // In drawing mode (not running, not paused): show default or custom surface overlay
    if (!runningRef.current && !pausedRef.current) {
      if (drawPointsRef.current.length > 1) {
        drawCustomSurface(ctx, drawPointsRef.current, brushSizeRef.current);
      } else if (!isClearedRef.current && !isDrawingRef.current) {
        // Show default surface only when not actively drawing
        if (surfaceTypeRef.current === 'flat') {
          drawFlatSurface(ctx, width, height);
        } else {
          drawDefaultSurface(ctx, width, height);
        }
      }
      // Do NOT draw simulation surface in drawing mode
    } else {
      // In simulation mode (running or paused): draw surface and balls
      // Draw custom surface if drawPoints exist, otherwise draw default/flat surface
      if (drawPointsRef.current.length > 1) {
        drawSimulationSurface(ctx, drawPointsRef.current);
      } else if (!isClearedRef.current) {
        // Draw surface during simulation based on surfaceType
        if (surfaceTypeRef.current === 'flat') {
          const flatSurface = [
            { x: 100, y: 450 },
            { x: 1100, y: 450 }
          ];
          drawSimulationSurface(ctx, flatSurface);
        } else {
          const defaultSurface = [
            { x: 300, y: 350 },
            { x: 600, y: 450 },
            { x: 900, y: 350 }
          ];
          drawSimulationSurface(ctx, defaultSurface);
        }
      }
      if ((runningRef.current || pausedRef.current) && ballsRef.current && ballsRef.current.length > 0) {
        drawBalls(ctx, ballsRef.current);
      }
    }
    
    if (runningRef.current) {
      animationFrameRef.current = requestAnimationFrame(render);
    }
  }, [drawGrid, drawDefaultSurface, drawFlatSurface, drawCustomSurface, drawSimulationSurface, drawBalls]);

  // Render when drawing state changes
  useEffect(() => {
    render();
  }, [drawPoints, brushSize, isCleared, surfaceType, render]);


  // Start/stop animation loop when running state changes
  useEffect(() => {
    if (runningRef.current) {
      animationFrameRef.current = requestAnimationFrame(render);
    } else {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      // Render once to show surface when stopped/paused
      render();
    }
  }, [isRunning, isPaused, render]);

  // Setup canvas and animation loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (runningRef.current) {
      animationFrameRef.current = requestAnimationFrame(render);
    }
    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, [render]);

  const handleMouseDown = useCallback((e) => {
    const isRunning = runningRef.current;
    const isPaused = pausedRef.current;
    
    if (isRunning || isPaused) return;
    
    setIsDrawing(true);
    const pos = getCanvasPosition(e);
    window.dispatchEvent(new CustomEvent('unifiedCanvasDrawPoint', { detail: pos }));
  }, [getCanvasPosition, setIsDrawing]);

  const handleMouseMove = useCallback((e) => {
    if (!isDrawingRef.current || runningRef.current || pausedRef.current) return;
    const pos = getCanvasPosition(e);
    window.dispatchEvent(new CustomEvent('unifiedCanvasDrawPoint', { detail: pos }));
  }, [getCanvasPosition]);

  const handleMouseUp = useCallback(() => {
    setIsDrawing(false);
  }, [setIsDrawing]);

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
