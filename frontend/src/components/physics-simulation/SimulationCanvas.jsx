import { useRef, useEffect } from 'react';
import './SimulationCanvas.css';

const SimulationCanvas = ({ ballsRef, surface, isRunning }) => {
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);
  const surfaceRef = useRef([]);
  const runningRef = useRef(false);

  // Keep refs in sync
  useEffect(() => {
    surfaceRef.current = surface || [];
  }, [surface]);

  useEffect(() => {
    runningRef.current = isRunning;
  }, [isRunning]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    console.log('[Canvas] Canvas size:', canvas.width, 'x', canvas.height);

    const render = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw surface
      if (surfaceRef.current && surfaceRef.current.length >= 2) {
        drawSurface(ctx, surfaceRef.current);
      }

      // Draw balls from ref (no re-render)
      if (ballsRef.current && ballsRef.current.length > 0) {
        drawBalls(ctx, ballsRef.current);
      }

      if (runningRef.current) {
        animationFrameRef.current = requestAnimationFrame(render);
      }
    };

    // Draw immediately
    render();

    // Start continuous rendering loop if running
    if (runningRef.current) {
      animationFrameRef.current = requestAnimationFrame(render);
      console.log('[Canvas] Animation loop started');
    } else {
      console.log('[Canvas] Animation loop NOT started, runningRef:', runningRef.current);
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [ballsRef, surface, isRunning]);

  const drawSurface = (ctx, surface) => {
    if (surface.length < 2) return;

    // Fill surface area
    ctx.fillStyle = 'rgba(100, 200, 255, 0.08)';
    ctx.beginPath();
    ctx.moveTo(surface[0].x, surface[0].y);
    for (let i = 1; i < surface.length; i++) {
      ctx.lineTo(surface[i].x, surface[i].y);
    }
    ctx.closePath();
    ctx.fill();

    // Draw surface line
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

    // Glow effect
    ctx.shadowColor = 'rgba(100, 200, 255, 0.5)';
    ctx.shadowBlur = 10;
    ctx.stroke();
    ctx.shadowBlur = 0;
  };

  const drawBalls = (ctx, balls) => {
    const activeBalls = balls.filter(b => b.active);
    
    console.log('[Canvas] Drawing', activeBalls.length, 'active balls');
    if (activeBalls.length > 0) {
      console.log('[Canvas] Ball coords:', activeBalls[0].x, activeBalls[0].y, 'radius:', activeBalls[0].radius);
    }
    
    // Batch draw by color for performance
    const ballsByColor = new Map();
    activeBalls.forEach(ball => {
      const key = `${ball.r},${ball.g},${ball.b}`;
      if (!ballsByColor.has(key)) {
        ballsByColor.set(key, []);
      }
      ballsByColor.get(key).push(ball);
    });

    // Draw each color group
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
    
    // DEBUG: Draw a large red ball at center to verify rendering
    ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
    ctx.beginPath();
    ctx.arc(600, 300, 20, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = 'white';
    ctx.font = '14px Arial';
    ctx.fillText('DEBUG - If you see this, rendering works!', 450, 270);
  };

  return (
    <canvas
      ref={canvasRef}
      width={1200}
      height={600}
      className="simulation-canvas"
    />
  );
};

export default SimulationCanvas;
