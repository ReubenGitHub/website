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
  isBallPaintingEnabled,
  spawnPixels,
  setSpawnPixels,
  defaultSurface,
  flatSurface,
  onGetSpawnMask
}) => {
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);
  const spawnMaskCanvasRef = useRef(null); // Offscreen canvas for spawn area painting
  const spawnPixelsRef = useRef(null); // Ref to current spawn pixels for render access (null = default, [] = cleared, array = custom)
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
  const isBallPaintingEnabledRef = useRef(isBallPaintingEnabled);
  const isBallPaintingRef = useRef(false); // Track if mouse is pressed during ball painting
  const paintingBoundsRef = useRef({ minX: 0, maxX: 0 }); // Bounding box during active painting
  const currentStrokeRef = useRef([]); // Batched points during active stroke
  const connectorEndRef = useRef(null); // End point of connector line (from old surface to new stroke start)

  // Generate SVG data URL cursor for ball painting mode
  const generateCursorSvg = useCallback((brushSize) => {
    const size = Math.max(brushSize * 2, 2); // 2x brush size, min 2px
    const half = size / 2;
    const strokeWidth = 2;
    const radius = Math.max(half - strokeWidth, 0.5);
    const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='${size}' height='${size}'><circle cx='${half}' cy='${half}' r='${radius}' fill='none' stroke='white' stroke-width='${strokeWidth}'/></svg>`;
    return `url("data:image/svg+xml,${encodeURIComponent(svg)}") ${half} ${half}, auto`;
  }, []);

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
  useEffect(() => { isBallPaintingEnabledRef.current = isBallPaintingEnabled; }, [isBallPaintingEnabled]);
  useEffect(() => { spawnPixelsRef.current = spawnPixels; }, [spawnPixels]);

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
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.stroke();
  }, []);

  // Draw on spawn mask canvas (offscreen)
  const paintSpawnArea = useCallback((pos) => {
    const maskCanvas = spawnMaskCanvasRef.current;
    if (!maskCanvas) return;
    const ctx = maskCanvas.getContext('2d');
    const radius = brushSizeRef.current;
    ctx.fillStyle = 'white';
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
    ctx.fill();
    
    // Update painting bounds live (account for brush radius)
    const bounds = paintingBoundsRef.current;
    if (pos.x - radius < bounds.minX) bounds.minX = pos.x - radius;
    if (pos.x + radius > bounds.maxX) bounds.maxX = pos.x + radius;
    
    console.log('[Canvas] paintSpawnArea: painted at', pos, 'radius:', radius, 'bounds:', bounds);
  }, []);

  // Generate default spawn area rectangle pixels (20% width/height, centered, 25% from top)
  const getDefaultSpawnPixels = useCallback(() => {
    const maskCanvas = spawnMaskCanvasRef.current;
    if (!maskCanvas) return [];
    const width = maskCanvas.width;
    const height = maskCanvas.height;
    const rectWidth = width * 0.2;
    const rectHeight = height * 0.2;
    const rectX = (width - rectWidth) / 2;
    const rectY = height * 0.25;
    const pixels = [];
    const step = 4;
    for (let y = rectY; y < rectY + rectHeight; y += step) {
      for (let x = rectX; x < rectX + rectWidth; x += step) {
        pixels.push({ x: Math.round(x), y: Math.round(y) });
      }
    }
    console.log('[Canvas] getDefaultSpawnPixels: generated', pixels.length, 'pixels for default rectangle');
    return pixels;
  }, []);

  // Extract painted pixel coordinates from spawn mask
  const extractSpawnPixels = useCallback(() => {
    const maskCanvas = spawnMaskCanvasRef.current;
    if (!maskCanvas) return [];
    const ctx = maskCanvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    const pixels = [];
    // Sample every 2nd pixel for better thin-stroke coverage
    const step = 2;
    for (let y = 0; y < maskCanvas.height; y += step) {
      for (let x = 0; x < maskCanvas.width; x += step) {
        const index = (y * maskCanvas.width + x) * 4;
        if (imageData.data[index] > 128) { // White pixel = painted
          pixels.push({ x, y });
        }
      }
    }
    console.log('[Canvas] extractSpawnPixels: found', pixels.length, 'painted pixels');
    return pixels;
  }, []);

  // Get raw spawn mask image data as byte array (1 byte per pixel: 0=empty, 255=painted)
  const getSpawnMaskData = useCallback(() => {
    const maskCanvas = spawnMaskCanvasRef.current;
    if (!maskCanvas) return null;
    const ctx = maskCanvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    const bytes = new Uint8Array(imageData.data.length / 4);
    for (let i = 0; i < bytes.length; i++) {
      bytes[i] = imageData.data[i * 4 + 3] > 128 ? 255 : 0; // Use alpha channel
    }
    return Array.from(bytes);
  }, []);

  // Draw spawn mask overlay on main canvas with rainbow gradient
  const drawSpawnMaskOverlay = useCallback((ctx, maskCanvas) => {
    // Check if we have spawnPixels (null = default, [] = cleared, array = custom)
    const currentSpawnPixels = spawnPixelsRef.current;
    const hasSpawnArea = currentSpawnPixels !== null && currentSpawnPixels.length > 0;
    
    // Save context state
    ctx.save();
    
    if (hasSpawnArea || isBallPaintingRef.current) {
      // Custom painted area: draw rainbow gradient using mask as stencil
      // Requires mask canvas to exist
      if (!maskCanvas) {
        ctx.restore();
        return;
      }
      
      ctx.save();
      
      // Draw the mask canvas first (white pixels = painted areas)
      ctx.drawImage(maskCanvas, 0, 0);
      
      // Get bounding box: use live painting bounds during painting, otherwise from spawnPixels
      let minX, maxX;
      if (isBallPaintingRef.current) {
        minX = paintingBoundsRef.current.minX;
        maxX = paintingBoundsRef.current.maxX;
      } else {
        // Calculate from spawnPixels (fast, no canvas scanning)
        minX = maskCanvas.width;
        maxX = 0;
        for (let i = 0; i < currentSpawnPixels.length; i++) {
          if (currentSpawnPixels[i].x < minX) minX = currentSpawnPixels[i].x;
          if (currentSpawnPixels[i].x > maxX) maxX = currentSpawnPixels[i].x;
        }
      }
      
      // Now draw gradient with source-in - keeps only pixels overlapping with mask
      // Gradient spans from left edge to right edge of painted area
      const gradient = ctx.createLinearGradient(minX, 0, maxX, 0);
      gradient.addColorStop(0.00, 'rgba(255, 0, 0, 1)');
      gradient.addColorStop(0.17, 'rgba(255, 127, 0, 1)');
      gradient.addColorStop(0.33, 'rgba(255, 255, 0, 1)');
      gradient.addColorStop(0.50, 'rgba(0, 255, 0, 1)');
      gradient.addColorStop(0.67, 'rgba(0, 0, 255, 1)');
      gradient.addColorStop(0.83, 'rgba(75, 0, 130, 1)');
      gradient.addColorStop(1.00, 'rgba(148, 0, 211, 1)');
      ctx.fillStyle = gradient;
      ctx.globalCompositeOperation = 'source-in';
      ctx.globalAlpha = 0.2;
      ctx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
      
      ctx.restore();
    } else if (currentSpawnPixels === null) {
      // Default spawn area: draw rainbow gradient rectangle
      // Does NOT require mask canvas - uses main canvas dimensions
      const width = ctx.canvas.width;
      const height = ctx.canvas.height;
      const rectWidth = width * 0.2;
      const rectHeight = height * 0.2;
      const rectX = (width - rectWidth) / 2;
      const rectY = height * 0.25;
      
      ctx.globalAlpha = 0.2;
      const gradient = ctx.createLinearGradient(rectX, 0, rectX + rectWidth, 0);
      gradient.addColorStop(0.00, 'rgba(255, 0, 0, 1)');
      gradient.addColorStop(0.17, 'rgba(255, 127, 0, 1)');
      gradient.addColorStop(0.33, 'rgba(255, 255, 0, 1)');
      gradient.addColorStop(0.50, 'rgba(0, 255, 0, 1)');
      gradient.addColorStop(0.67, 'rgba(0, 0, 255, 1)');
      gradient.addColorStop(0.83, 'rgba(75, 0, 130, 1)');
      gradient.addColorStop(1.00, 'rgba(148, 0, 211, 1)');
      ctx.fillStyle = gradient;
      ctx.fillRect(rectX, rectY, rectWidth, rectHeight);
    }
    
    ctx.globalAlpha = 1.0;
    ctx.restore();
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
    
    // Draw spawn mask overlay (bottom layer)
    drawSpawnMaskOverlay(ctx, spawnMaskCanvasRef.current);

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
    
    // Draw grid on top of everything (spawn mask, surface, balls) for consistent visibility
    drawGrid(ctx, width, height);
    
    if (runningRef.current) {
      animationFrameRef.current = requestAnimationFrame(render);
    }
  }, [drawGrid, drawDefaultSurface, drawFlatSurface, drawCustomSurface, drawStroke, drawSimulationSurface, drawBalls]);

  // Render when drawing state changes
  useEffect(() => {
    console.log('[Canvas] useEffect[drawPoints] firing - drawPoints:', drawPoints.length, 'isDrawingRef:', isDrawingRef.current, 'strokeLen:', currentStrokeRef.current.length);
    render(drawPoints);
  }, [drawPoints, brushSize, isCleared, surfaceType, render]);

  // Clear spawn mask canvas when spawnPixels is null (default) or [] (cleared)
  useEffect(() => {
    if (spawnPixels && spawnPixels.length > 0) return;
    const maskCanvas = spawnMaskCanvasRef.current;
    if (!maskCanvas) return;
    const ctx = maskCanvas.getContext('2d');
    ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    // Reset painting bounds when clearing
    paintingBoundsRef.current = { minX: 0, maxX: 0 };
    console.log('[Canvas] useEffect[spawnPixels]: cleared spawn mask canvas (spawnPixels:', spawnPixels === null ? 'null' : '[]', ')');
  }, [spawnPixels]);

  // Re-render when spawnPixels changes (to update spawn area overlay)
  useEffect(() => {
    console.log('[Canvas] useEffect[spawnPixels] firing - spawnPixels:', spawnPixels === null ? 'default' : spawnPixels.length);
    render(drawPointsRef.current);
  }, [spawnPixels, render]);


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

  // Setup canvas and offscreen spawn mask canvas
  useEffect(() => {
    console.log('[Canvas] useEffect[canvas setup] firing - running:', runningRef.current);
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    // Create offscreen canvas for spawn mask
    if (!spawnMaskCanvasRef.current) {
      const maskCanvas = document.createElement('canvas');
      maskCanvas.width = canvas.width;
      maskCanvas.height = canvas.height;
      spawnMaskCanvasRef.current = maskCanvas;
      console.log('[Canvas] Spawn mask canvas created:', canvas.width, 'x', canvas.height);
    }
    
    if (runningRef.current) {
      animationFrameRef.current = requestAnimationFrame(render);
    }
    return () => {
      console.log('[Canvas] useEffect[canvas setup] cleanup');
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, [render]);

  const handleMouseDown = useCallback((e) => {
    const pos = getCanvasPosition(e);
    console.log('[Canvas] mousedown - pos:', pos, 'ballPainting:', isBallPaintingEnabledRef.current, 'surfaceDrawing:', isSurfaceDrawingEnabledRef.current);
    
    if (runningRef.current) { console.log('[Canvas] mousedown BLOCKED: isRunning'); return; }
    
    // Ball painting mode
    if (isBallPaintingEnabledRef.current) {
      console.log('[Canvas] mousedown: ball painting mode');
      isBallPaintingRef.current = true;
      // Initialize or expand bounds based on whether we have existing painted pixels
      const currentSpawnPixels = spawnPixelsRef.current;
      const hasExistingPixels = currentSpawnPixels && currentSpawnPixels.length > 0;
      if (!hasExistingPixels) {
        // Fresh start after clear: initialize bounds to start position
        paintingBoundsRef.current = { minX: pos.x, maxX: pos.x };
      } else {
        // Expand bounds to include new start point (preserve total painted width)
        const bounds = paintingBoundsRef.current;
        if (pos.x < bounds.minX) bounds.minX = pos.x;
        if (pos.x > bounds.maxX) bounds.maxX = pos.x;
      }
      paintSpawnArea(pos);
      render();
      return;
    }
    
    // Surface drawing mode
    if (!isSurfaceDrawingEnabledRef.current) { console.log('[Canvas] mousedown BLOCKED: drawing not enabled'); return; }
    
    // Clear existing surface when starting a new drawing session
    const prevStroke = currentStrokeRef.current;
    if (prevStroke.length > 0) {
      console.log('[Canvas] mousedown: committing prev stroke, points:', prevStroke.length);
      setDrawPoints(prev => [...prev, ...prevStroke]);
      currentStrokeRef.current = [];
    }
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
  }, [getCanvasPosition, setIsDrawing, setDrawPoints, drawPoints, render, paintSpawnArea]);

  const handleMouseMove = useCallback((e) => {
    const pos = getCanvasPosition(e);
    
    // Ball painting mode - only paint when mouse button is pressed
    if (isBallPaintingEnabledRef.current && !runningRef.current && isBallPaintingRef.current) {
      paintSpawnArea(pos);
      render();
      return;
    }
    
    if (!isDrawingRef.current || runningRef.current) { console.log('[Canvas] mousemove BLOCKED - drawing:', isDrawingRef.current, 'running:', runningRef.current); return; }
    if (!isSurfaceDrawingEnabledRef.current) { console.log('[Canvas] mousemove BLOCKED: not enabled'); return; }
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
    // Reset ball painting state
    isBallPaintingRef.current = false;
    
    // Extract spawn pixels if ball painting was active
    if (isBallPaintingEnabledRef.current) {
      const pixels = extractSpawnPixels();
      console.log('[Canvas] mouseup: extracted', pixels.length, 'spawn pixels');
      setSpawnPixels(pixels);
    }
    
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
  }, [setDrawPoints, setIsDrawing, extractSpawnPixels, setSpawnPixels]);

  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <canvas
        ref={canvasRef}
        width={1200}
        height={600}
        className="simulation-canvas"
        style={isBallPaintingEnabled ? { cursor: generateCursorSvg(brushSize) } : {}}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => {
          setIsDrawing(false);
        }}
      />
    </div>
  );
};

export default UnifiedCanvas;
