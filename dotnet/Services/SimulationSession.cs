using DotnetApi.Models;

namespace DotnetApi.Services;

public class SimulationSession : IDisposable
{
    private readonly SurfaceService _surfaceService;
    private readonly ILogger<SimulationSession> _logger;
    private Task? _simulationTask;
    private CancellationTokenSource? _cancellationTokenSource;
    private readonly object _stateLock = new();
    private Ball[] _balls = Array.Empty<Ball>();
    private int _canvasWidth = 1200;
    private int _canvasHeight = 600;

    public SimulationConfig Config { get; private set; } = new();
    public bool IsRunning { get; private set; }

    public SimulationSession(SurfaceService surfaceService, ILogger<SimulationSession> logger)
    {
        _surfaceService = surfaceService;
        _logger = logger;
    }

    public void SetConfig(SimulationConfig config)
    {
        Config = config;
        _canvasWidth = 1200;
        _canvasHeight = 600;
    }

    public void SetSurface(List<SurfacePoint> surface)
    {
        _surfaceService.SetSurface(surface);
    }

    public void SetCanvasSize(int width, int height)
    {
        _canvasWidth = width;
        _canvasHeight = height;
    }

    public void Start()
    {
        lock (_stateLock)
        {
            if (IsRunning) return;
            IsRunning = true;
            _cancellationTokenSource = new CancellationTokenSource();
            var ct = _cancellationTokenSource.Token;
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
            _cancellationTokenSource?.Cancel();
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
            return new SimulationState { Balls = _balls, Timestamp = DateTime.UtcNow.Ticks };
        }
    }

    private Ball[] SpawnBalls(int count)
    {
        var balls = new Ball[count];
        var surface = _surfaceService.Surface;
        if (surface.Count < 2)
        {
            for (int i = 0; i < count; i++)
            {
                var x = 100 + (i % 100) * 10;
                var y = 50 + (i / 100) * 10;
                balls[i] = CreateBall(x, y);
            }
            return balls;
        }
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
        var r = (byte)(normalizedX * 255);
        var g = (byte)(255 - Math.Abs(normalizedX - 0.5) * 2 * 255);
        var b = (byte)((1 - normalizedX) * 255);
        return new Ball(x, y, (Random.Shared.NextDouble() - 0.5) * 100, 0, 2.5, r, g, b);
    }

    private async Task SimulationLoop(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested && IsRunning)
        {
            try
            {
                var gravity = Config.Gravity * 100;
                var deltaTime = Config.DeltaTime;
                var restitution = Config.Restitution;
                var airResistance = Config.AirResistance;
                Parallel.For(0, _balls.Length, i =>
                {
                    if (!_balls[i].Active) return;
                    var ball = _balls[i];
                    ball.Vy += gravity * deltaTime;
                    ball.Vx *= (1 - airResistance);
                    ball.Vy *= (1 - airResistance);
                    ball.X += ball.Vx * deltaTime;
                    ball.Y += ball.Vy * deltaTime;
                    CheckSurfaceCollision(ref ball, restitution);
                    if (ball.X < -50 || ball.X > _canvasWidth + 50 || ball.Y < -50 || ball.Y > _canvasHeight + 50)
                    {
                        ball.Active = false;
                    }
                    _balls[i] = ball;
                });
            }
            catch (OperationCanceledException) { break; }
            catch (Exception ex) { _logger.LogError(ex, "Error in simulation loop"); break; }
            await Task.Delay((int)(Config.DeltaTime * 1000), ct);
        }
    }

    private void CheckSurfaceCollision(ref Ball ball, double restitution)
    {
        var surface = _surfaceService.Surface;
        if (surface.Count < 2) return;
        for (int i = 0; i < surface.Count - 1; i++)
        {
            var p1 = surface[i];
            var p2 = surface[i + 1];
            var dx = p2.X - p1.X;
            var dy = p2.Y - p1.Y;
            var lengthSq = dx * dx + dy * dy;
            if (lengthSq == 0) continue;
            var t = Math.Max(0, Math.Min(1, ((ball.X - p1.X) * dx + (ball.Y - p1.Y) * dy) / lengthSq));
            var closestX = p1.X + t * dx;
            var closestY = p1.Y + t * dy;
            var distX = ball.X - closestX;
            var distY = ball.Y - closestY;
            var distance = Math.Sqrt(distX * distX + distY * distY);
            if (distance < ball.Radius)
            {
                var normalX = distX / distance;
                var normalY = distY / distance;
                var dotProduct = ball.Vx * normalX + ball.Vy * normalY;
                ball.Vx = (ball.Vx - 2 * dotProduct * normalX) * restitution;
                ball.Vy = (ball.Vy - 2 * dotProduct * normalY) * restitution;
                ball.X = closestX + normalX * (ball.Radius + 0.1);
                ball.Y = closestY + normalY * (ball.Radius + 0.1);
            }
        }
    }

    public void Dispose()
    {
        Stop();
        _cancellationTokenSource?.Dispose();
    }
}
