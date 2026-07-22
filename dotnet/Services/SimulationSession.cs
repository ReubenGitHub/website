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
    private bool _isResetting = false;

    public SimulationConfig Config { get; private set; } = new();
    public bool IsRunning { get; private set; }
    private CancellationTokenSource? _streamingCts;

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

    public void SetRestitution(double restitution)
    {
        Config.Restitution = Math.Max(0, Math.Min(1, restitution));
        _logger.LogInformation("Restitution updated to {Restitution}", Config.Restitution);
    }

    public void Start()
    {
        _logger.LogInformation("Start() called on session {SessionId}", System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(this));
        lock (_stateLock)
        {
            _isResetting = false;
            // Wait for old task to finish before disposing (prevents ObjectDisposedException)
            if (_simulationTask != null && !_simulationTask.IsCompleted)
            {
                _cancellationTokenSource?.Cancel();
                try { _simulationTask.Wait(TimeSpan.FromSeconds(2)); }
                catch (AggregateException) { /* Task was cancelled, ignore */ }
                _cancellationTokenSource?.Dispose();
            }
            
            _cancellationTokenSource = new CancellationTokenSource();
            var ct = _cancellationTokenSource.Token;
            
            IsRunning = true;
            
            if (_balls.Length == 0 || _balls.All(b => !b.Active))
            {
                _balls = SpawnBalls(Config.BallCount, Config.SpawnPixels);
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
            _logger.LogInformation("Simulation paused (physics held, streaming stopped)");
        }
    }

    public void Resume()
    {
        _logger.LogInformation("Simulation resumed");
        lock (_stateLock)
        {
            IsRunning = true;
            _logger.LogInformation("Simulation loop resumed (balls preserved)");
        }
    }

    public void Reset()
    {
        lock (_stateLock)
        {
            _isResetting = true;
            if (_simulationTask != null && !_simulationTask.IsCompleted)
            {
                _cancellationTokenSource?.Cancel();
                try { _simulationTask.Wait(TimeSpan.FromSeconds(2)); }
                catch (AggregateException) { /* Task was cancelled, ignore */ }
                _cancellationTokenSource?.Dispose();
            }
            _cancellationTokenSource?.Dispose();
            _cancellationTokenSource = null;
            _simulationTask = null;
            _streamingCts?.Cancel();
            _streamingCts = null;
            _balls = Array.Empty<Ball>();
            IsRunning = false;
            _logger.LogInformation("Simulation reset");
        }
    }

    public void Stop()
    {
        lock (_stateLock)
        {
            _isResetting = true;
            if (_simulationTask != null && !_simulationTask.IsCompleted)
            {
                _cancellationTokenSource?.Cancel();
                try { _simulationTask.Wait(TimeSpan.FromSeconds(2)); }
                catch (AggregateException) { /* Task was cancelled, ignore */ }
                _cancellationTokenSource?.Dispose();
            }
            _cancellationTokenSource?.Dispose();
            _cancellationTokenSource = null;
            _simulationTask = null;
            _streamingCts?.Cancel();
            _streamingCts = null;
            _balls = Array.Empty<Ball>();
            IsRunning = false;
            _isResetting = false;
        }
    }

    public SimulationState GetState()
    {
        lock (_stateLock)
        {
            return new SimulationState { Balls = _balls, Timestamp = DateTime.UtcNow.Ticks };
        }
    }

    private Ball[] SpawnBalls(int count, List<SpawnPoint>? spawnPixels)
    {
        var balls = new Ball[count];
        
        _logger.LogInformation("SpawnBalls called: spawnPixels is null={IsNull}, count={Count}", spawnPixels == null, spawnPixels?.Count ?? 0);
        
        // Use spawn pixels if provided
        if (spawnPixels != null && spawnPixels.Count > 0)
        {
            _logger.LogInformation("Spawning {Count} balls in custom spawn area ({PixelCount} pixels)", count, spawnPixels.Count);
            // Find spawn area bounds
            var spawnMinX = spawnPixels.Min(p => p.X);
            var spawnMaxX = spawnPixels.Max(p => p.X);
            var spawnMinY = spawnPixels.Min(p => p.Y);
            var spawnMaxY = spawnPixels.Max(p => p.Y);
            
            // Stratified sampling: divide area into cells, pick one random point per cell
            var spawnPoints = CreateStratifiedSpawnPoints(spawnMinX, spawnMaxX, spawnMinY, spawnMaxY, count);
            for (int i = 0; i < count; i++)
            {
                var point = spawnPoints[i];
                balls[i] = CreateBall(point.X, point.Y, spawnMinX, spawnMaxX);
            }
            return balls;
        }
        
        // Fallback to surface-based spawning
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

    /// <summary>
    /// Creates uniformly distributed spawn points using stratified sampling.
    /// Divides the area into cells and picks one random point per cell for even coverage.
    /// </summary>
    private List<(double X, double Y)> CreateStratifiedSpawnPoints(double minX, double maxX, double minY, double maxY, int count)
    {
        var result = new List<(double X, double Y)>(count);
        var width = maxX - minX;
        var height = maxY - minY;
        
        // Calculate grid dimensions: aim for roughly square cells
        var aspectRatio = width / Math.Max(1, height);
        var cols = (int)Math.Ceiling(Math.Sqrt(count * aspectRatio));
        var rows = (int)Math.Ceiling((double)count / cols);
        var cellWidth = width / cols;
        var cellHeight = height / rows;
        
        // One random point per cell for uniform coverage
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                if (result.Count >= count) break;
                
                var px = minX + col * cellWidth + Random.Shared.NextDouble() * cellWidth;
                var py = minY + row * cellHeight + Random.Shared.NextDouble() * cellHeight;
                result.Add((px, py));
            }
        }
        
        return result;
    }

    private Ball CreateBall(double x, double y)
    {
        // Fallback: use surface bounds for coloring
        var surface = _surfaceService.Surface;
        var minX = surface.Any() ? surface.Min(p => p.X) : 100;
        var maxX = surface.Any() ? surface.Max(p => p.X) : 700;
        return CreateBallWithBounds(x, y, minX, maxX);
    }

    private Ball CreateBall(double x, double y, double spawnMinX, double spawnMaxX)
    {
        return CreateBallWithBounds(x, y, spawnMinX, spawnMaxX);
    }

    private Ball CreateBallWithBounds(double x, double y, double minX, double maxX)
    {
        // Inverted gradient: left=violet, right=red (opposite of frontend)
        var normalizedX = (x - minX) / Math.Max(1, maxX - minX);
        var inv = 1.0 - normalizedX; // invert direction
        var r = (byte)(inv * 255);
        var g = (byte)(255 - Math.Abs(inv - 0.5) * 2 * 255);
        var b = (byte)((1 - inv) * 255);
        return new Ball(x, y, 0, 0, 2.5, r, g, b);
    }

    private async Task SimulationLoop(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try
            {
                // Only update physics if running (paused = skip physics, preserve state)
                if (IsRunning && !_isResetting)
                {
                    // Capture array reference to avoid race conditions
                    var currentBalls = _balls;
                    var surface = _surfaceService.Surface;
                    
                    // Use PhysicsEngine with sub-stepping
                    var engine = new PhysicsEngine(Config, surface, _canvasWidth, _canvasHeight);
                    var state = engine.Update(currentBalls);
                    
                    // Update balls array after physics processing completes
                    lock (_stateLock)
                    {
                        if (_isResetting) break;
                        _balls = state.Balls;
                    }
                }
            }
            catch (OperationCanceledException) { break; }
            catch (Exception ex) { _logger.LogError(ex, "Error in simulation loop"); break; }
            await Task.Delay((int)(Config.DeltaTime * 1000), ct);
        }
    }

    public CancellationTokenSource? GetOrCreateStreamingCts()
    {
        lock (_stateLock)
        {
            if (_streamingCts == null || _streamingCts.IsCancellationRequested)
            {
                _streamingCts = new CancellationTokenSource();
                _logger.LogInformation("Created new streaming CTS for session {SessionId}", System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(this));
            }
            return _streamingCts;
        }
    }

    public void CancelStreaming()
    {
        lock (_stateLock)
        {
            _streamingCts?.Cancel();
        }
    }

    public void Dispose()
    {
        Stop();
    }
}
