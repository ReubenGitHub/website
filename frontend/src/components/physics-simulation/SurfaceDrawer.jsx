import { useState, useRef, useEffect } from 'react';
import './SurfaceDrawer.css';

const SurfaceDrawer = ({ onSurfaceDrawn, defaultSurface }) => {
  const [isDrawing, setIsDrawing] = useState(false);
  const [brushSize, setBrushSize] = useState(8);
  const [points, setPoints] = useState([]);
  const [useDefault, setUseDefault] = useState(true);
  const canvasRef = useRef(null);
  const containerRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !containerRef.current) return;

    const resizeCanvas = () => {
      const rect = containerRef.current.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = 400;
      drawCanvas();
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    return () => window.removeEventListener('resize', resizeCanvas);
  }, [points, useDefault, brushSize]);

  const drawCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let x = 0; x < canvas.width; x += 50) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += 50) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    if (useDefault && points.length === 0) {
      drawDefaultSurface(ctx, canvas);
    } else if (points.length > 1) {
      drawCustomSurface(ctx);
    }
  };

  const drawDefaultSurface = (ctx, canvas) => {
    const centerX = canvas.width / 2;
    const bottomY = canvas.height - 50;

    ctx.strokeStyle = 'rgba(100, 200, 255, 0.8)';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    ctx.beginPath();
    ctx.moveTo(centerX - 250, bottomY);
    ctx.lineTo(centerX, bottomY + 50);
    ctx.lineTo(centerX + 250, bottomY);
    ctx.stroke();

    // Fill surface area
    ctx.fillStyle = 'rgba(100, 200, 255, 0.05)';
    ctx.beginPath();
    ctx.moveTo(centerX - 250, bottomY);
    ctx.lineTo(centerX, bottomY + 50);
    ctx.lineTo(centerX + 250, bottomY);
    ctx.closePath();
    ctx.fill();
  };

  const drawCustomSurface = (ctx) => {
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

    // Draw brush points
    ctx.fillStyle = 'rgba(100, 200, 255, 0.3)';
    points.forEach(point => {
      ctx.beginPath();
      ctx.arc(point.x, point.y, brushSize / 2, 0, Math.PI * 2);
      ctx.fill();
    });
  };

  const getCanvasPosition = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
  };

  const handleMouseDown = (e) => {
    if (useDefault) return;
    setIsDrawing(true);
    const pos = getCanvasPosition(e);
    const newPoints = [...points, pos];
    setPoints(newPoints);
  };

  const handleMouseMove = (e) => {
    if (!isDrawing || useDefault) return;
    const pos = getCanvasPosition(e);
    // Only add point if distance from last point is sufficient
    if (points.length > 0) {
      const lastPoint = points[points.length - 1];
      const dist = Math.sqrt((pos.x - lastPoint.x) ** 2 + (pos.y - lastPoint.y) ** 2);
      if (dist < 5) return;
    }
    setPoints([...points, pos]);
  };

  const handleMouseUp = () => {
    setIsDrawing(false);
  };

  const handleClear = () => {
    setPoints([]);
    setUseDefault(true);
  };

  const handleUseDefault = () => {
    setUseDefault(true);
    setPoints([]);
  };

  const handleDrawMode = () => {
    setUseDefault(false);
  };

  const handleSubmit = () => {
    if (useDefault) {
      // Use default surface
      const canvas = canvasRef.current;
      const centerX = canvas.width / 2;
      const bottomY = canvas.height - 50;
      const surfacePoints = [
        { x: centerX - 250, y: bottomY },
        { x: centerX, y: bottomY + 50 },
        { x: centerX + 250, y: bottomY }
      ];
      onSurfaceDrawn(surfacePoints);
    } else if (points.length >= 2) {
      onSurfaceDrawn(points);
    }
  };

  return (
    <div ref={containerRef} className="surface-drawer-container">
      <div className="surface-drawer-controls">
        <div className="control-group">
          <label>Brush Size: {brushSize}px</label>
          <input
            type="range"
            min="4"
            max="20"
            value={brushSize}
            onChange={(e) => setBrushSize(Number(e.target.value))}
            disabled={useDefault}
          />
        </div>
        <div className="control-buttons">
          <button
            className={`btn ${useDefault ? 'active' : ''}`}
            onClick={handleUseDefault}
          >
            Default Surface
          </button>
          <button
            className={`btn ${!useDefault ? 'active' : ''}`}
            onClick={handleDrawMode}
          >
            Draw Surface
          </button>
          <button className="btn btn-secondary" onClick={handleClear}>
            Clear
          </button>
          <button className="btn btn-primary" onClick={handleSubmit}>
            Use Surface
          </button>
        </div>
      </div>
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        className="surface-drawer-canvas"
      />
      <p className="drawer-hint">
        {useDefault
          ? 'Using default right-angle corner surface (45° up-left and up-right)'
          : points.length > 0
            ? `Drawn ${points.length} points. Click "Use Surface" when ready.`
            : 'Click and drag to draw your surface'}
      </p>
    </div>
  );
};

export default SurfaceDrawer;
