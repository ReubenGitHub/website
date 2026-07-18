# 2D Physics Simulation — Plan

## Overview

A 2D physics simulation where users draw a surface, spawn thousands of balls above it, and watch them bounce independently. Ball trajectories are fully parallelized on the .NET backend, with positions streamed to the React frontend for rendering.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  FRONTEND (React — Vite)                                │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Surface     │  │ Ball Spawn   │  │ Canvas 2D     │  │
│  │ Drawer      │  │ Controls     │  │ Renderer      │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
│         │                 │                 │           │
│         └─────────────────┼─────────────────┘           │
│                           │ SignalR (WebSocket)         │
└───────────────────────────┼─────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────┐
│  BACKEND (.NET 8 — ASP.NET Core)                        │
│                           │                             │
│  ┌────────────────────────┼──────────────────────────┐  │
│  │ Physics Engine (Parallel.For)                     │  │
│  │  • Gravity application                            │  │
│  │  • Surface collision detection                    │  │
│  │  • Bounce/restitution calculation                 │  │
│  │  • Air resistance (optional)                      │  │
│  └────────────────────────┼──────────────────────────┘  │
│                           │                             │
│  ┌────────────────────────┼──────────────────────────┐  │
│  │ SignalR Hub                                           │  │
│  │  • Stream ball positions (30-60fps)                 │  │
│  │  • Handle commands (start, pause, reset, spawn)     │  │
│  └────────────────────────┼──────────────────────────┘  │
│                           │                             │
│  ┌────────────────────────┼──────────────────────────┐  │
│  │ Surface Storage (in-memory)                         │  │
│  │  • Store user-drawn surface as polyline             │  │
│  └────────────────────────┼──────────────────────────┘  │
└───────────────────────────┼─────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────┐
│  DOCKER COMPOSE                                           │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────┐  │
│  │ Frontend    │    │ .NET API     │    │ Existing   │  │
│  │ (port 3000) │    │ + SignalR    │    │ ML API     │  │
│  │             │    │ (port 5001)  │    │ (port 5000)│  │
│  └─────────────┘    └──────────────┘    └────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## File Structure

```
dotnet/
├── Controllers/
│   └── SimulationController.cs          # REST endpoints (config, presets)
├── Hubs/
│   └── PhysicsHub.cs                    # SignalR hub for real-time streaming
├── Services/
│   ├── PhysicsEngine.cs                 # Core simulation engine (parallel)
│   ├── SurfaceService.cs                # Surface storage/validation
│   └── SimulationSession.cs             # Per-user simulation state
├── Models/
│   ├── Ball.cs                          # Ball state (struct for performance)
│   ├── SurfacePoint.cs                  # Surface polyline point
│   ├── SimulationConfig.cs              # Gravity, restitution, ball count, etc.
│   └── SimulationState.cs               # Full state for streaming
├── Middleware/
│   └── SimulationCleanupMiddleware.cs   # Clean up abandoned sessions
├── Program.cs                           # SignalR registration, CORS
├── appsettings.json
└── dotnet-api.csproj

frontend/src/components/
├── physics-simulation/
│   ├── PhysicsSimulation.jsx            # Main page component
│   ├── PhysicsSimulation.css            # Page styles
│   ├── SurfaceDrawer.jsx                # Surface drawing component
│   ├── SurfaceDrawer.css
│   ├── SpawnControls.jsx                # Ball spawn controls
│   ├── SpawnControls.css
│   ├── SimulationCanvas.jsx             # Canvas rendering component
│   ├── SimulationCanvas.css
│   ├── SimulationControls.jsx           # Play/pause/reset/speed
│   ├── SimulationControls.css
│   └── PhysicsSimulation.css            # Component styles
├── physics-simulation.css               # Page-specific styles
└── physics-simulation.jsx               # Re-export stub
```

---

## Physics Engine Design

### Ball Model
```csharp
public struct Ball
{
    public double X;          // Position (pixels)
    public double Y;          // Position (pixels)
    public double Vx;         // Velocity X
    public double Vy;         // Velocity Y
    public double Radius;     // Ball radius
    public byte R, G, B;      // Color (for visual variety)
    public bool Active;       // Is ball still in simulation?
}
```

### Surface Model
- Stored as array of `(X, Y)` points forming a polyline (from freehand draw)
- Collision detection: find which surface segment the ball is above, calculate distance to segment
- If distance < ball radius → collision → reflect velocity with restitution
- Default surface: right-angle corner (two lines at 45° meeting at bottom center)

### Ball Coloring (Gradient)
- When balls are spawned, each ball's color is determined by its spawn position
- Gradient applied across the drawn area (e.g., left=blue, center=green, right=red, or similar)
- Color calculated once at spawn time, stored in ball struct
- Visual effect: balls appear as a colored cloud matching the gradient of the draw area

### Simulation Loop (Backend)
```csharp
// Per frame:
1. For each active ball (Parallel.For):
   a. Apply gravity: Vy += gravity * deltaTime
   b. Update position: X += Vx * deltaTime, Y += Vy * deltaTime
   c. Check surface collision:
      - Find nearest surface segment
      - If ball intersects surface:
        * Calculate collision normal
        * Reflect velocity: V = V - 2(V·N)N
        * Apply restitution: V *= restitution
        * Push ball out of surface
   d. Check bounds (off-screen?): mark inactive or wrap
2. Collect all ball positions
3. Stream to frontend via SignalR
```

### Parallelization Strategy
- Each ball's update is **independent** (no ball-ball collisions)
- `Parallel.For` spreads work across all CPU cores
- Use `Span<Ball>` or `ArrayPool<Ball>` to avoid GC pressure
- Lock-free state collection using `Parallel.For` with local buffers

---

## Frontend Design

### Canvas Rendering
- HTML5 Canvas 2D — simple, fast enough for 10,000 balls at 30fps
- Draw surface as filled polygon with dark gradient (matches site theme)
- Draw balls as small circles (radius ~2-3px, treated as points)
- Balls colored with gradient based on draw area position
- `requestAnimationFrame` for smooth 30fps rendering (synced to backend updates)

### Interaction Flow
1. **Draw Surface**: Freehand draw with paintbrush tool (circle size adjustable)
   - Default surface: right-angle corner (45° up-left, 45° up-right)
   - Drawn area defines the containment boundary
2. **Set Ball Count**: Slider 10-10,000 (default 2,000)
3. **Start Simulation**: Click "Start" → balls evenly spread in drawn area → SignalR connection → simulation begins
4. **Controls**: Pause, reset, change ball count (requires reset)
5. **Balls disappear** when they go off-screen

### State Management
- Surface points stored in React state
- Ball positions received from SignalR, stored in ref (not state — avoids re-renders)
- Canvas renders from ref data directly

---

## SignalR Protocol

### Client → Server Commands
```
{ type: 'start', config: { ballCount, gravity, restitution, ... } }
{ type: 'pause' }
{ type: 'resume' }
{ type: 'reset' }
{ type: 'spawn', spawnArea: { x, y, width, height } }
{ type: 'surface', points: [{x, y}, ...] }
{ type: 'config', config: { ... } }
```

### Server → Client Updates
```
{ type: 'state', balls: [{x, y, r, color}, ...], timestamp: 123456 }
```

### Connection Management
- One SignalR connection per simulation session
- Session ID in URL: `/physics-simulation?session=abc123`
- Server cleans up abandoned sessions after timeout

---

## Key Design Decisions (CONFIRMED)

### User Requirements:

1. **Ball count**: 2,000 default, slider 10-10,000 (backend validation enforces limits)

2. **Surface drawing**: Freehand draw with paintbrush/circle size tool.
   - Once simulation starts, balls are evenly spread through the drawn area
   - Default surface: right-angle corner (45° up-left and up-right edges)

3. **Ball colors**: Each ball gets a color based on a gradient applied to the drawn area

4. **No presets** — user-drawn surface only

5. **Page URL**: `/physics-simulation`

6. **No real-time parameter adjustment** (gravity, restitution, etc. are fixed)

7. **Ball sizes**: All same size, small enough to be treated as points

8. **Balls disappear** when they go off-screen

9. **Backend**: Part of existing .NET service (port 5001)

10. **Target FPS**: 30fps

11. **Rendering**: Canvas 2D (simple, fast enough for 10,000 balls)

12. **Theme**: Match existing dark glassmorphism theme, colored balls with gradient

---

## Implementation Phases

### Phase 1: Foundation
- [ ] Create .NET project structure (Hubs, Services, Models)
- [ ] Implement SignalR hub
- [ ] Create basic physics engine (gravity + surface collision)
- [ ] Single-threaded version first
- [ ] Docker compose integration

### Phase 2: Parallelization
- [ ] Convert to `Parallel.For`
- [ ] Optimize with `Span<T>`, `ArrayPool`
- [ ] Benchmark single vs parallel performance
- [ ] Add session management

### Phase 3: Frontend — Drawing
- [ ] Surface drawer component (click to place points)
- [ ] Canvas rendering (surface + balls)
- [ ] Spawn controls UI

### Phase 4: Frontend — Simulation
- [ ] SignalR connection management
- [ ] Real-time ball rendering from streamed data
- [ ] Play/pause/reset controls
- [ ] Speed multiplier

### Phase 5: Polish
- [ ] Preset scenarios
- [ ] Visual polish (colors, trails, gradients)
- [ ] Responsive design
- [ ] Performance optimization
- [ ] Add to navbar and home page

### Phase 6: Deployment
- [ ] Docker compose update (add physics service)
- [ ] AWS deployment config
- [ ] Documentation

---

## Performance Considerations

### Backend
- `Parallel.For` with degree of parallelism = CPU core count
- `Span<Ball>` for zero-allocation updates
- `ArrayPool<Ball>.Shared` for reusing ball arrays
- Avoid GC during simulation loop (pre-allocate all objects)
- Target: 10,000 balls at 30fps on t4g.small

### Frontend
- Canvas 2D `beginPath()` / `fill()` batching
- `requestAnimationFrame` for rendering
- Store ball positions in `Float32Array` for efficient canvas drawing
- Avoid React state updates during simulation (use refs)
- Target: 10,000 balls at 60fps on modern laptop

### Network
- SignalR binary protocol (MessagePack) instead of JSON
- Compress ball data (quantize coordinates to short integers)
- Target payload: ~20KB per frame for 5,000 balls
- At 30fps = 600KB/s (acceptable for most connections)
