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
│  DEVCONTAINER (single container — VS Code tasks)        │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────┐  │
│  │ Frontend    │    │ .NET API     │    │ Existing   │  │
│  │ (port 3000) │    │ + SignalR    │    │ ML API     │  │
│  │             │    │ (port 5001)  │    │ (port 5000)│  │
│  └─────────────┘    └──────────────┘    └────────────┘  │
└─────────────────────────────────────────────────────────┘

Note: All services run inside a single devcontainer (python:3.9-slim base).
Services start via VS Code tasks on folder open, not Docker compose.
Production deployment files moved to /app/deployment/.
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
│   └── (empty — cleanup middleware not yet implemented)
├── Program.cs                           # SignalR registration, CORS, Serilog logging
├── dotnet-api.csproj                    # Includes Serilog.AspNetCore, Serilog.Sinks.File
└── appsettings.json

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
- One SignalR connection per simulation session (scoped via DI)
- Server cleans up abandoned sessions after timeout (not yet implemented)

---

## Current Status

### Working
- ✅ dotnet service builds and runs on port 5001
- ✅ SignalR hub connects (WebSocket transport, confirmed working)
- ✅ Frontend component renders at `/physics-simulation`
- ✅ Surface drawing UI functional
- ✅ Spawn controls UI functional
- ✅ REST endpoint `/api/example/hello` responds
- ✅ CORS configured for localhost:3000 + [::1]:3000
- ✅ Serilog file logging to `/app/logs/dotnet-.log`
- ✅ Task output logging to `/app/logs/task-{flask,dotnet,react}.log` (auto-cleared on restart)
- ✅ SignalR parameter passing bug fixed (StartSimulation takes separate config + surface args)
- ✅ `_streamingCts` cancellation bug fixed (fresh CTS created in StartSimulation)
- ✅ DI scoping bug fixed (GetOrCreateSession stores session in Context.Items)
- ✅ ObjectDisposedException fixed (SimulationStreamService BackgroundService using IHubContext)
- ✅ Clients.User → Clients.Client fix (send to connection ID, not user claim)
- ✅ WebSocket transport confirmed working (LongPolling had issues through Vite proxy)
- ✅ Streaming loop continues during pause (sends frozen positions), restarts on resume
- ✅ Pause() sets IsRunning=false (physics loop skips updates, preserves state)
- ✅ Resume() sets IsRunning=true (physics loop resumes immediately from frozen state)
- ✅ ResumeSimulation hub method accepts config + surface params (safety net)
- ✅ Frontend StateUpdate handler ignores empty balls arrays (defensive)
- ✅ SimulationSession registered as singleton (persists across all hub invocations)
- ✅ ResetSimulation calls _streamService.StopStreaming()
- ✅ PhysicsHub methods: StartSimulation, PauseSimulation, ResumeSimulation, ResetSimulation all functional
- ✅ DI registration fixed (SimulationStreamService → singleton, SimulationSession → singleton, SurfaceService → singleton)
- ✅ Session cleanup on disconnect added (OnDisconnectedAsync disposes session + scope)
- ✅ JSON serialization fixed (camelCase for SignalR to match frontend property names)
- ✅ Frontend uses refs for ball data (not React state) — avoids re-render overhead
- ✅ Canvas renders continuously with requestAnimationFrame (not only on state change)
- ✅ Canvas re-renders when surface changes (surface added to useEffect deps)
- ✅ Default surface auto-selected on mount
- ✅ Surface collision detection already implemented in PhysicsEngine/SimulationSession
- ✅ Ball spawning with gradient coloring already implemented in SpawnBalls/CreateBall
- ✅ Use Surface button works — surface appears in simulation canvas
- ✅ StartSimulation invocation reaches backend with correct payload
- ✅ SignalR connection retry mechanism added (handles HMR race conditions)
- ✅ Animation loop starts correctly when isRunning changes (isRunning added to useEffect deps)
- ✅ Balls spawn, stream, and animate visibly in canvas
- ✅ Pause/Resume/Reset controls work correctly with reconnection logic
- ✅ Start() waits for old task before creating new (prevents ObjectDisposedException)
- ✅ Frontend ensureConnected() helper for pause/resume/reset (auto-reconnects if needed)

### Previously Blocked (Now Resolved)
- ✅ StartSimulation DI registration — resolved with full restart + connection retry
- ✅ Animation loop not starting — fixed by adding `isRunning` to SimulationCanvas useEffect deps

### Known Issues
1. **Hot reload doesn't apply DI changes** — must fully restart dotnet task when Program.cs DI registration changes
2. **No session timeout cleanup** — abandoned sessions persist until explicit disconnect
3. **Canvas responsive scaling** — fixed: uses CSS `aspect-ratio: 2/1` to maintain proportions (balls stay round)
4. **React Strict Mode double connections** — accepted as dev-only behavior (1 connection in production)
5. **Edge balls may fall through** — ~38/2000 balls near surface edges fall through (1.9%), all others (98.1%) stay on surface

### Recent Fixes
- ✅ **Restitution parameter now configurable** — added slider (0-1, step 0.01) to SpawnControls. Backend default is 0.4. Frontend was previously sending hardcoded 0.7 (mismatch). Now uses user-selected value passed to backend in StartSimulation/ResumeSimulation.
- ✅ **Clear Drawing + simulation surface bug** — clicking "Clear Drawing" sets isCleared=true, but drawing new points didn't reset it, causing handlePlay to pass null/empty surface to backend (balls fell through). Added useEffect to auto-reset isCleared when drawPoints.length > 0.
- ✅ **Surface invisible during simulation** — UnifiedCanvas drew surfaceRef.current which was empty after removing surface prop. Now draws drawPoints or default surface during simulation mode.
- ✅ **Animation loop not starting** — added `isRunning` to useEffect deps in SimulationCanvas
- ✅ **SignalR connection fails after HMR** — added retry mechanism with exponential backoff
- ✅ **Canvas aspect ratio distortion** — replaced fixed `height: 600px` with `aspect-ratio: 2/1`
- ✅ **Double SignalR connection on mount** — accepted as expected Strict Mode behavior (only 1 connection in prod)
- ✅ **Resume resets simulation** — switched to singleton session + ManualResetEventSlim pause event (physics loop blocks instead of cancelling, preserving all ball positions/velocities)
- ✅ **Session persistence** — SimulationSession registered as singleton, GetOrCreateSession resolves directly (no more scoped DI session recreation)
- ✅ **Balls frozen during pause** — removed ManualResetEventSlim blocking, physics loop checks IsRunning flag (skips physics when paused, continues loop)
- ✅ **Resume didn't restart streaming** — StreamingLoop no longer breaks on IsRunning=false, continues with frozen positions
- ✅ **Subsequent Start() calls failed** — Start() waits up to 2s for old task before creating new (prevents ObjectDisposedException)
- ✅ **Resume sometimes fails with "not connected" error** — added ensureConnected() helper to pause/resume/reset (auto-reconnects via SignalR start())
- ✅ **Sub-stepping** — 8 sub-steps per frame prevents tunneling
- ✅ **Velocity clamping** — MaxDistancePerStep = 1.5px prevents balls from moving too far per sub-step
- ✅ **Resting state detection** — balls settle when velocity is low and near surface
- ✅ **Physics parameters tuned** — gravity=4.0, restitution=0.4, airResistance=0.03
- ✅ **CRITICAL: Balls falling through surface** — fixed inverted dotProduct condition (`if (dotProduct > 0)` → `if (dotProduct < 0)`) in PhysicsEngine.cs CheckSurfaceCollision. Ball count improved from 122-142/2000 (6-7%) to 1962/2000 (98.1%)

### Ball Spawn Area Painting (In Progress)
- ✅ **Backend model updated** — SimulationConfig now accepts `List<SpawnPoint>` for custom spawn pixels
- ✅ **Backend spawning logic** — CreateEvenlySpacedSpawnPoints uses grid-based approach for even distribution in arbitrary pixel area
- ✅ **Frontend state** — spawnPixels state (null=default, []=cleared, array=custom), isBallPaintingEnabled state added to PhysicsSimulation
- ✅ **Offscreen spawn mask canvas** — UnifiedCanvas uses offscreen canvas for pixel-based painting
- ✅ **Paint brush rendering** — spawn mask drawn as semi-transparent blue overlay (rgba(100, 150, 255, 0.15))
- ✅ **Pixel extraction** — extractSpawnPixels samples every 4th pixel for performance
- ✅ **Default spawn area rectangle** — 20% width/height, centered horizontally, 25% from top. Visible as semi-transparent blue overlay. Set by default button.
- ✅ **Default spawn area visible** — rendered on canvas when spawnPixels is null (default state)
- ✅ **Ball button styling fixed** — color updated from #8892b0 to #e0e0e0 to match surface buttons
- ✅ **Clear button always enabled** — removed spawnPixels.length check, clear is a noop when empty
- ✅ **Default spawn button sets rectangle** — calls handleDefaultSpawnArea which sets spawnPixels to null
- ⏳ **Mutual exclusivity** — ball painting and surface drawing toggles are mutually exclusive (working)
- ⏳ **Painting blocked during simulation** — only works when simulation stopped (working)

### Next Steps
1. ✅ Test StartSimulation — balls spawn, stream, and animate (VERIFIED)
2. ✅ Test Pause/Resume/Reset controls (VERIFIED — resume preserves ball state, reconnection works)
3. ✅ UI streamlined — single Play button (section 3), single Reset button (section 3), Remove button removed from section 2
5. Add session cleanup middleware (timeout-based)
6. Performance optimization (ArrayPool, Span<T>)
7. Visual polish (trails, glow effects, responsive design)
8. ✅ Unit tests added (42 tests, all passing) — SimulationSession, SurfaceService, Ball, SimulationConfig
9. ✅ **All 42 tests passing** — updated test assertions to match new default physics values (gravity=4.0, restitution=0.4, airResistance=0.03) and corrected default surface geometry (X=100 instead of X=400)
10. ✅ **Unified canvas implemented** — merged SurfaceDrawer and SimulationCanvas into single UnifiedCanvas component. Drawing mode active ONLY when simulation stopped (not paused). When paused, balls remain visible (simulation mode). Canvas shows default V-shape or custom drawn surface when stopped. HTML overlay controls (dropdown, brush slider, clear, use surface) above canvas. Canvas size: 1200x600.
11. ✅ **Canvas UX simplified** — removed default/custom mode dropdown. Canvas always starts with default V-shape visible, user can immediately draw without switching modes. Added "Use Default Surface" button (clears drawing, shows default) and "Clear Drawing" button (clears canvas completely). Removed "Use Surface" button — surface is auto-used when "Play" is clicked (drawPoints if available, otherwise defaultSurface). Coordinate alignment fixed (scaleX/scaleY to account for CSS scaling). Simulation mode now correctly draws surface (drawPoints or default) instead of empty surfaceRef. isCleared state auto-resets when user starts drawing.
12. ✅ **Clear Drawing + simulation surface bug fixed** — clicking "Clear Drawing" sets isCleared=true, but drawing new points didn't reset it, causing handlePlay to pass null surface to backend. Added useEffect to reset isCleared when drawPoints.length > 0. Surface now correctly sent to backend after clearing and redrawing.
13. ✅ **Restitution (bounciness) parameter added** — backend default is 0.4. Added restitution slider (0-1, step 0.01) to SpawnControls UI. Value passed to backend in StartSimulation and ResumeSimulation hub calls. Can be changed at any time — updates apply on next play/resume.

---

## Testing Strategy

### Unit Tests ✅ COMPLETE (42 tests)
**Location**: `/app/tests/dotnet.Tests/`
**Framework**: xUnit + coverlet collector
**Status**: All 42 tests passing

| Test File | Tests | What's Covered |
|---|---|---|
| `SimulationSessionTests.cs` | 20 | Start/Pause/Resume/Reset lifecycle, state preservation, ball spawning, multiple cycles, dispose |
| `SurfaceServiceTests.cs` | 9 | SetSurface, AddPoint, Clear, IsValid, GetDefaultSurface, null handling |
| `BallTests.cs` | 6 | Constructor, properties, array storage, negative velocity |
| `SimulationConfigTests.cs` | 7 | Default values, settable properties, independence |

### Integration Tests (Deferred)
**Priority: MEDIUM** — Worth adding after Phase 6 (deployment)

| Test Target | What to Test |
|---|---|
| SignalR Hub | StartSimulation → StateUpdate stream → Pause → Resume → Reset |
| SimulationController | REST endpoints (validate config, presets) |
| Full lifecycle | Client connects → starts → streams → disconnects |

**Framework**: xUnit + WebApplicationFactory<T> + InMemory messaging

### E2E Tests (Deferred)
**Priority: LOW** — Complex to set up, low ROI until UI is final

**What to test**: Full user flow (draw surface → start → pause → resume → reset)
**Framework**: Playwright (already available in devcontainer)
**When**: After visual polish is complete

### Integration Tests (Deferred)
**Priority: MEDIUM** — Worth adding after Phase 6 (deployment)

| Test Target | What to Test |
|---|---|
| SignalR Hub | StartSimulation → StateUpdate stream → Pause → Resume → Reset |
| SimulationController | REST endpoints (validate config, presets) |
| Full lifecycle | Client connects → starts → streams → disconnects |

**Framework**: xUnit + WebApplicationFactory<T> + InMemory messaging

### E2E Tests (Deferred)
**Priority: LOW** — Complex to set up, low ROI until UI is final

**What to test**: Full user flow (draw surface → start → pause → resume → reset)
**Framework**: Playwright (already available in devcontainer)
**When**: After visual polish is complete

### Recommendation
**Add unit tests NOW** for `SimulationSession` and `SurfaceService` — these are the most critical and stable components. Integration tests can wait until after deployment is working.

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

### Phase 1: Foundation ✅ COMPLETE
- [x] Create .NET project structure (Hubs, Services, Models)
- [x] Implement SignalR hub (`PhysicsHub.cs`)
- [x] Create basic physics engine (gravity + surface collision) — single-threaded
- [x] Session management (`SimulationSession.cs`)
- [x] Devcontainer setup (all 3 services running via VS Code tasks)
- [x] Serilog file logging configured
- [x] Task output logging configured (`/app/logs/task-{name}.log`)
- [x] Session cleanup on disconnect implemented
- [ ] Middleware for session cleanup (timeout-based, not yet implemented)

### Phase 2: Parallelization
- [x] Parallel.For implemented in SimulationLoop
- [ ] Optimize with `Span<T>`, `ArrayPool` (future)
- [ ] Benchmark single vs parallel performance (future)

### Phase 3: Frontend — Drawing & Canvas ✅ COMPLETE
- [x] Unified canvas component (`UnifiedCanvas.jsx`) — merges SurfaceDrawer + SimulationCanvas
- [x] Drawing mode when simulation stopped/paused — shows default V-shape or custom drawn surface
- [x] Simulation mode when running — shows surface + animated balls
- [x] Canvas controls UI — "Use Default Surface" button, "Clear Drawing" button, brush slider
- [x] No mode switching needed — canvas starts with default surface, user can draw immediately
- [x] Auto-use surface on play — whatever is on canvas when "Play" clicked is used (drawPoints or defaultSurface)
- [x] Spawn controls UI (`SpawnControls.jsx`)
- [x] Simulation controls UI (`SimulationControls.jsx`)

### Phase 4: Frontend — Simulation ✅ COMPLETE
- [x] SignalR connection management (WebSocket transport, connection works)
- [x] Real-time ball rendering from streamed data (VERIFIED working — 1958/2000 balls visible)
- [x] Unified Play/Pause/Reset controls (Play handles both start and resume)
- [x] Ball data stored in refs (not state) — no re-render overhead
- [x] Canvas renders continuously with requestAnimationFrame
- [x] Connection retry mechanism for HMR resilience
- [x] Animation loop correctly starts on isRunning change
- [ ] Speed multiplier (future)

### Phase 5: Polish
- [ ] Preset scenarios
- [ ] Visual polish (colors, trails, gradients)
- [ ] Responsive design
- [ ] Performance optimization
- [x] Add to navbar and home page (page exists at `/physics-simulation`)

### Phase 5: Polish
- [ ] Preset scenarios
- [ ] Visual polish (colors, trails, gradients)
- [ ] Responsive design
- [ ] Performance optimization
- [x] Add to navbar and home page (page exists at `/physics-simulation`)

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
