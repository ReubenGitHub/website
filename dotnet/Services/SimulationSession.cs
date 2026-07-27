using DotnetApi.Models;
using System.IO.Compression;
using System.IO;

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
                _balls = SpawnBalls(Config.BallCount, Config.SpawnPixels, Config.SpawnMask);
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

    private Ball[] SpawnBalls(int count, List<SpawnPoint>? spawnPixels, List<byte>? spawnMask)
    {
        var balls = new Ball[count];
        
        _logger.LogInformation("SpawnBalls called: spawnPixels is null={IsNull}, mask is null={MaskNull}, count={Count}", spawnPixels == null, spawnMask == null, count);
        
        // Decompress spawn mask if compressed (prefix byte 67 = 'C')
        byte[]? rawMaskBytes = null;
        if (spawnMask != null && spawnMask.Count > 0 && spawnMask[0] == 67) // 'C' = compressed
        {
            _logger.LogInformation("Spawn mask is compressed ({CompressedSize} bytes), decompressing...", spawnMask.Count - 1);
            try
            {
                var compressedBytes = spawnMask.Skip(1).ToArray();
                using var decompressedStream = new MemoryStream();
                using var deflateStream = new DeflateStream(new MemoryStream(compressedBytes), CompressionMode.Decompress);
                deflateStream.CopyTo(decompressedStream);
                rawMaskBytes = decompressedStream.ToArray();
                _logger.LogInformation("Spawn mask decompressed to {DecompressedSize} bytes", rawMaskBytes.Length);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to decompress spawn mask, using raw data");
                rawMaskBytes = spawnMask.ToArray();
            }
        }
        else if (spawnMask != null && spawnMask.Count > 0)
        {
            rawMaskBytes = spawnMask.ToArray();
        }
        
        // Use spawn mask if provided (higher precision than pixel list)
        if (rawMaskBytes != null && rawMaskBytes.Length > 0)
        {
            _logger.LogInformation("Spawning {Count} balls from spawn mask ({MaskSize} bytes)", count, rawMaskBytes.Length);
            const int mWidth = 1200;
            const int mHeight = 600;
            var spawnPoints = CreateSpawnPointsFromMask(rawMaskBytes.ToList(), count);
            // Compute actual bounds from painted pixels for correct coloring
            var spawnMinX = (double)mWidth; // Will find min
            var spawnMaxX = 0.0; // Will find max
            for (int i = 0; i < rawMaskBytes.Length; i++)
            {
                if (rawMaskBytes[i] > 128)
                {
                    var px = i % mWidth;
                    var py = i / mWidth;
                    if (px < spawnMinX) spawnMinX = px;
                    if (px > spawnMaxX) spawnMaxX = px;
                }
            }
            spawnMinX -= 0.5;
            spawnMaxX += 0.5;
            for (int i = 0; i < count; i++)
            {
                var point = spawnPoints[i];
                balls[i] = CreateBall(point.X, point.Y, spawnMinX, spawnMaxX);
            }
            return balls;
        }
        
        // Use spawn pixels if provided (fallback to pixel-based method)
        if (spawnPixels != null && spawnPixels.Count > 0)
        {
            _logger.LogInformation("Spawning {Count} balls in custom spawn area ({PixelCount} pixels)", count, spawnPixels.Count);
            // Find spawn area bounds
            var spawnMinX = spawnPixels.Min(p => p.X);
            var spawnMaxX = spawnPixels.Max(p => p.X);
            var spawnMinY = spawnPixels.Min(p => p.Y);
            var spawnMaxY = spawnPixels.Max(p => p.Y);
            
            // Stratified sampling from painted pixels only (respects shape, not bounding box)
            var spawnPoints = CreateStratifiedSpawnPoints(spawnPixels, count);
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
    /// Creates uniformly distributed spawn points using Bridson's Poisson Disk algorithm.
    /// This eliminates the grid structure that causes moiré effects in jittered grid approaches.
    /// Uses minimum-distance-based sampling with painted pixel validation.
    /// Research-backed: Red Blob Games confirms Poisson Disk is the gold standard for
    /// uniform point distribution with no grid patterns, no clustering, and no gaps.
    /// </summary>
    private List<(double X, double Y)> CreateStratifiedSpawnPoints(List<SpawnPoint> pixels, int count)
    {
        var result = new List<(double X, double Y)>(count);
        if (pixels.Count == 0 || count == 0) return result;
        
        // Collect unique painted pixel positions
        var paintedPixels = new HashSet<(int X, int Y)>();
        foreach (var pixel in pixels)
        {
            paintedPixels.Add(((int)pixel.X, (int)pixel.Y));
        }
        
        if (paintedPixels.Count == 0) return result;
        
        var minX = pixels.Min(p => p.X);
        var maxX = pixels.Max(p => p.X);
        var minY = pixels.Min(p => p.Y);
        var maxY = pixels.Max(p => p.Y);
        var spawnWidth = maxX - minX;
        var spawnHeight = maxY - minY;
        
        // Shuffle-based approach: Fisher-Yates shuffle ensures every painted pixel has
        // an equal chance of being selected, with NO grid structure and MAXIMUM coverage.
        // When count <= paintedPixels.Count, each ball gets a unique pixel (no duplicates).
        // When count > paintedPixels.Count, we cycle through the shuffled list.
        var pixelList = new List<(int X, int Y)>(paintedPixels);
        
        // Fisher-Yates shuffle (partial - only need first 'count' elements)
        int shuffleLimit = Math.Min(count, pixelList.Count);
        for (int i = 0; i < shuffleLimit; i++)
        {
            int j = Random.Shared.Next(i, pixelList.Count);
            (pixelList[i], pixelList[j]) = (pixelList[j], pixelList[i]);
        }
        
        for (int i = 0; i < count; i++)
        {
            var pixel = pixelList[i % pixelList.Count];
            // Full pixel jitter: ±0.5px ensures uniform coverage within the pixel
            result.Add((pixel.X + 0.5 + (Random.Shared.NextDouble() - 0.5),
                       pixel.Y + 0.5 + (Random.Shared.NextDouble() - 0.5)));
        }
        return result;
    }

    /// <summary>
    /// Creates spawn points directly from the spawn mask using Bridson's Poisson Disk algorithm.
    /// Same approach as CreateStratifiedSpawnPoints but works with raw mask byte array.
    /// </summary>
    private List<(double X, double Y)> CreateSpawnPointsFromMask(List<byte> mask, int count)
    {
        var result = new List<(double X, double Y)>(count);
        if (mask.Count == 0 || count == 0) return result;
        
        const int maskWidth = 1200;
        const int maskHeight = 600;
        
        // Collect painted pixels and find bounds in single pass
        var paintedPixels = new List<(int Px, int Py)>();
        var minX = (double)maskWidth;
        var maxX = 0.0;
        var minY = (double)maskHeight;
        var maxY = 0.0;
        
        for (int py = 0; py < maskHeight; py++)
        {
            for (int px = 0; px < maskWidth; px++)
            {
                var idx = py * maskWidth + px;
                if (idx < mask.Count && mask[idx] > 128)
                {
                    paintedPixels.Add((px, py));
                    if (px < minX) minX = px;
                    if (px > maxX) maxX = px;
                    if (py < minY) minY = py;
                    if (py > maxY) maxY = py;
                }
            }
        }
        
        _logger.LogInformation("Spawn mask: found {PaintedPixels} painted pixels in area [{minX},{minY}]-[{maxX},{maxY}]", 
            paintedPixels.Count, minX, minY, maxX, maxY);
        
        if (paintedPixels.Count == 0) return result;
        
        var spawnWidth = maxX - minX;
        var spawnHeight = maxY - minY;
        
        // Handle degenerate cases
        if (spawnWidth <= 0 || spawnHeight <= 0)
        {
            for (int i = 0; i < count; i++)
            {
                var pixel = paintedPixels[Random.Shared.Next(paintedPixels.Count)];
                result.Add((pixel.Px + 0.5 + (Random.Shared.NextDouble() - 0.5), 
                           pixel.Py + 0.5 + (Random.Shared.NextDouble() - 0.5)));
            }
            return result;
        }
        
        // Shuffle-based approach: Fisher-Yates shuffle ensures every painted pixel has
        // an equal chance of being selected, with NO grid structure and MAXIMUM coverage.
        // When count <= paintedPixels.Count, each ball gets a unique pixel (no duplicates).
        // When count > paintedPixels.Count, we cycle through the shuffled list.
        
        // Fisher-Yates shuffle (partial - only need first 'count' elements)
        int shuffleLimit = Math.Min(count, paintedPixels.Count);
        for (int i = 0; i < shuffleLimit; i++)
        {
            int j = Random.Shared.Next(i, paintedPixels.Count);
            (paintedPixels[i], paintedPixels[j]) = (paintedPixels[j], paintedPixels[i]);
        }
        
        for (int i = 0; i < count; i++)
        {
            var pixel = paintedPixels[i % paintedPixels.Count];
            // Full pixel jitter: ±0.5px ensures uniform coverage within the pixel
            result.Add((pixel.Px + 0.5 + (Random.Shared.NextDouble() - 0.5),
                       pixel.Py + 0.5 + (Random.Shared.NextDouble() - 0.5)));
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
        // Rainbow gradient: left=red, right=violet (same direction as frontend)
        // Uses HSV color space for proper rainbow colors ending in violet
        var normalizedX = (x - minX) / Math.Max(1.0, maxX - minX);
        var hue = 278.0 * normalizedX; // 0=red at left, 278=violet at right
        var (r, g, b) = HsvToRgb(hue, 1.0, 1.0);
        return new Ball(x, y, 0, 0, 2.5, r, g, b);
    }

    private static (byte R, byte G, byte B) HsvToRgb(double h, double s, double v)
    {
        var c = s * v;
        var x = c * (1 - Math.Abs((h / 60.0) % 2 - 1));
        var m = v - c;
        double r, g, b;

        if (h < 60) { r = c; g = x; b = 0; }
        else if (h < 120) { r = x; g = c; b = 0; }
        else if (h < 180) { r = 0; g = c; b = x; }
        else if (h < 240) { r = 0; g = x; b = c; }
        else if (h < 300) { r = x; g = 0; b = c; }
        else { r = c; g = 0; b = x; }

        return ((byte)((r + m) * 255), (byte)((g + m) * 255), (byte)((b + m) * 255));
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
