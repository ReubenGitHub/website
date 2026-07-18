using DotnetApi.Models;

namespace DotnetApi.Services;

public class SimulationSession : IDisposable
{
    private readonly SurfaceService _surfaceService;
    private readonly PhysicsEngine _physicsEngine;
    private readonly ILogger<SimulationSession> _logger;
    private Task? _simulationTask;
    private CancellationTokenSource? _cancellationTokenSource;
    private readonly object _stateLock = new();
    private Ball[] _balls = Array.Empty<Ball>();
    private int _canvasWidth = 800;
    private int _canvasHeight = 600;

    public SimulationConfig Config { get; private set; } = new();
    public bool IsRunning { get; private set; }

    public SimulationSession(SurfaceService surfaceService, ILogger<SimulationSession> logger)
    {
        _surfaceService = surfaceService;
        _logger = logger;
        _physicsEngine = new PhysicsEngine(Config, surfaceService.Surface, _canvasWidth, _canvasHeight);
    }

    public void SetConfig(SimulationConfig config)
    {
        Config = config;
        _canvasWidth = 1200;
        _canvasHeight = 600;
        _physicsEngine = new PhysicsEngine(Config, _surfaceService.Surface, _canvasWidth, _canvasHeight);
    }

    public void SetSurface(List<SurfacePoint> surface)
    {
        _surfaceService.SetSurface(surface);
    }

    public void SetCanvasSize(int width, int height)
    {
        _canvasWidth = width;
        _canvasHeight = height;
        _physicsEngine = new PhysicsEngine(Config, _surfaceService.Surface, _canvasWidth, _canvasHeight);
    }

    public void Start()
    {
        lock (_stateLock)
        {
            if (IsRunning) return;

            IsRunning = true;
            _cancellationTokenSource = new CancellationTokenSource();
            var ct = _cancellationTokenSource.Token;

            // Spawn balls if not already spawned
            if (_balls.Length == 0 || _balls.All(b => !b.Active))
            {
                _balls = SpawnBalls(Config.BallCount);
            }

            _simulationTask = Task.Run(() => SimulationLoop(ct), ct);
            _logger.LogInformation("Simulation started with {BallCount} balls", Config.BallCount);
        }
    }

    public void Pause()
    {
        lock (_stateLock)
        {
            IsRunning = false;
            _cancellationTokenSource?.Cancel();
            _logger.LogInformation("Simulation paused");
        }
    }

    public void Resume()
    {
        lock (_stateLock)
        {
            if (IsRunning) return;

            IsRunning = true;
            _cancellationTokenSource = new CancellationTokenSource();
            var ct = _cancellationTokenSource.Token;
            _simulationTask = Task.Run(() => SimulationLoop(ct), ct);
            _logger.LogInformation("Simulation resumed");
        }
    }

    public void Reset()
    {
        lock (_stateLock)
        {
            IsRunning = false;
            _cancellationTokenSource?.Cancel();
            _balls = Array.Empty<Ball>();
            _logger.LogInformation("Simulation reset");
        }
    }

    public void Stop()
    {
        lock (_stateLock)
        {
            IsRunning = false;
            _cancellationTokenSource?.Cancel();
            _balls = Array.Empty<Ball>();
        }
    }

    public SimulationState GetState()
    {
        lock (_stateLock)
        {
            return new SimulationState
            {
                Balls = _balls,
                Timestamp = DateTime.UtcNow.Ticks
            };
        }
    }

    private Ball[] SpawnBalls(int count)
    {
        var balls = new Ball[count];
        var surface = _surfaceService.Surface;

        if (surface.Count < 2)
        {
            // Use default spawn area if no surface defined
            for (int i = 0; i < count; i++)
            {
                var x = 100 + (i % 100) * 10;
                var y = 50 + (i / 100) * 10;
                balls[i] = CreateBall(x, y);
            }
            return balls;
        }

        // Find bounding box of surface
        var minX = surface.Min(p => p.X);
        var maxX = surface.Max(p => p.X);
        var minY = surface.Min(p => p.Y);

        var spawnWidth = maxX - minX;
        var spawnHeight = Math.Max(100, spawnWidth * 0.5);

        var cols = (int)Math.Ceiling(Math.Sqrt(count * spawnWidth / spawnHeight));
        var rows = (int)Math.Ceiling((double)count / cols);
        var spacingX = spawnWidth / cols;
        var spacingY = spawnHeight / rows;

        for (int i = 0; i < count; i++)
        {
            var col = i % cols;
            var row = i / cols;
            var x = minX + col * spacingX + spacingX / 2 + (Random.Shared.NextDouble() - 0.5) * spacingX * 0.5;
            var y = minY - row * spacingY - spacingY / 2 + (Random.Shared.NextDouble() - 0.5) * spacingY * 0.5;
            balls[i] = CreateBall(x, y);
        }

        return balls;
    }

    private Ball CreateBall(double x, double y)
    {
        var surface = _surfaceService.Surface;
        var minX = surface.Any() ? surface.Min(p => p.X) : 100;
        var maxX = surface.Any() ? surface.Max(p => p.X) : 700;
        var normalizedX = (x - minX) / Math.Max(1, maxX - minX);

        // Gradient: blue (left) -> green (center) -> red (right)
        var r = (byte)(normalizedX * 255);
        var g = (byte)(255 - Math.Abs(normalizedX - 0.5) * 2 * 255);
        var b = (byte)((1 - normalizedX) * 255);

        return new Ball(
            x: x,
            y: y,
            vx: (Random.Shared.NextDouble() - 0.5) * 100,
            vy: 0,
            radius: 2.5,
            r: r,
            g: g,
            b: b
        );
    }

    private async Task SimulationLoop(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested && IsRunning)
        {
            try
            {
                var state = _physicsEngine.Update(_balls);
                // State is retrieved via GetState() by the hub
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in simulation loop");
                break;
            }

            await Task.Delay((int)(Config.DeltaTime * 1000), ct);
        }
    }

    public void Dispose()
    {
        Stop();
        _cancellationTokenSource?.Dispose();
    }
}
