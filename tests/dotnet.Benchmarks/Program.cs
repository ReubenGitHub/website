using BenchmarkDotNet.Running;
using DotnetApi.Models;
using DotnetApi.Services;
using dotnet.Benchmarks;
using System.Diagnostics;

Console.WriteLine("=== Physics Simulation Benchmark & Profiler ===\n");

// ============================================================================
// PART 1: Detailed Profiling (breaks down time by operation)
// ============================================================================
Console.WriteLine("═══ PART 1: DETAILED PROFILING ═══\n");
await ProfileDetailed();

// ============================================================================
// PART 2: BenchmarkDotNet Statistical Benchmarking
// ============================================================================
Console.WriteLine("\n\n═══ PART 2: STATISTICAL BENCHMARKS ═══\n");
Console.WriteLine("Running BenchmarkDotNet benchmarks...\n");
var summary = BenchmarkDotNet.Running.BenchmarkRunner.Run<PhysicsEngineBenchmarks>();

async Task ProfileDetailed()
{
    var ballCounts = new[] { 100, 500, 1000, 2000, 5000 };
    
    foreach (var count in ballCounts)
    {
        Console.WriteLine($"\n--- Profiling with {count} balls ---");
        await ProfileWithCount(count);
    }
}

async Task ProfileWithCount(int ballCount)
{
    var config = new SimulationConfig
    {
        BallCount = ballCount,
        Gravity = 4.0,
        Restitution = 0.4,
        DeltaTime = 1.0 / 30.0
    };

    // Generate surfaces
    var flatSurface = GenerateFlatSurface();
    var hillySurface = GenerateHillySurface();
    
    var iterations = (ballCount <= 1000) ? 200 : 100;

    // Profile flat surface
    var flatEngine = new PhysicsEngine(config, flatSurface, 1200, 600);
    var flatBalls = SpawnBalls(ballCount, seed: 42);
    
    var flatStopwatch = Stopwatch.StartNew();
    for (int i = 0; i < iterations; i++)
    {
        var cloned = CloneBalls(flatBalls);
        flatEngine.Update(cloned);
    }
    flatStopwatch.Stop();
    
    Console.WriteLine($"  Flat surface:  {flatStopwatch.ElapsedMilliseconds / (double)iterations:F3} ms/frame | {flatStopwatch.ElapsedMilliseconds} ms total ({iterations} frames)");
    Console.WriteLine($"    Potential FPS: {1000.0 / (flatStopwatch.ElapsedMilliseconds / (double)iterations):F1}");

    // Profile hilly surface
    var hillyEngine = new PhysicsEngine(config, hillySurface, 1200, 600);
    var hillyBalls = SpawnBalls(ballCount, seed: 43);
    
    var hillyStopwatch = Stopwatch.StartNew();
    for (int i = 0; i < iterations; i++)
    {
        var cloned = CloneBalls(hillyBalls);
        hillyEngine.Update(cloned);
    }
    hillyStopwatch.Stop();
    
    Console.WriteLine($"  Hilly surface: {hillyStopwatch.ElapsedMilliseconds / (double)iterations:F3} ms/frame | {hillyStopwatch.ElapsedMilliseconds} ms total ({iterations} frames)");
    Console.WriteLine($"    Potential FPS: {1000.0 / (hillyStopwatch.ElapsedMilliseconds / (double)iterations):F1}");
    
    // Per-ball analysis
    var perBallMs = (flatStopwatch.ElapsedMilliseconds / (double)iterations) / ballCount * 1000;
    Console.WriteLine($"  Per-ball: {perBallMs:F2} µs/ball/frame");
    
    // Sub-step analysis (16 sub-steps per frame)
    var perSubStepMs = (flatStopwatch.ElapsedMilliseconds / (double)iterations) / ballCount / 16 * 1000000;
    Console.WriteLine($"  Per ball per sub-step: {perSubStepMs:F1} ns");
}

List<SurfacePoint> GenerateFlatSurface()
{
    var surface = new List<SurfacePoint>();
    for (double x = 100; x <= 1100; x += 3.0)
    {
        surface.Add(new SurfacePoint(x, 400));
    }
    return surface;
}

List<SurfacePoint> GenerateHillySurface()
{
    var surface = new List<SurfacePoint>();
    for (double x = 100; x <= 1100; x += 3.0)
    {
        var y = 400 + 50 * Math.Sin(x * 0.02);
        surface.Add(new SurfacePoint(x, y));
    }
    return surface;
}

Ball[] SpawnBalls(int count, int seed = 42)
{
    var balls = new Ball[count];
    var random = new Random(seed);
    for (int i = 0; i < count; i++)
    {
        double x = 150 + random.NextDouble() * 900;
        double y = 50 + random.NextDouble() * 200;
        balls[i] = new Ball(x, y, 0, 0, 2.5, 
            (byte)(random.Next(0, 256)), 
            (byte)(random.Next(0, 256)), 
            (byte)(random.Next(0, 256)));
    }
    return balls;
}

Ball[] CloneBalls(Ball[] source)
{
    var cloned = new Ball[source.Length];
    for (int i = 0; i < source.Length; i++)
    {
        cloned[i] = source[i];
    }
    return cloned;
}
