using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Order;
using DotnetApi.Models;
using DotnetApi.Services;
using Microsoft.Extensions.Logging.Abstractions;

namespace dotnet.Benchmarks;

[MemoryDiagnoser]
[ShortRunJob] // Minimal warmup + 5-9 measurement iterations (runs in ~1-2 min instead of 30+ min)
[Orderer(BenchmarkDotNet.Order.SummaryOrderPolicy.FastestToSlowest)]
[RankColumn]
public class PhysicsEngineBenchmarks
{
    private PhysicsEngine? _engine;
    private Ball[]? _balls;
    private List<SurfacePoint>? _flatSurface;
    private List<SurfacePoint>? _hillySurface;
    private List<SurfacePoint>? _complexSurface;

    // Different ball counts to test scaling
    [Params(100, 500, 1000, 2000, 5000)]
    public int BallCount { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var config = new SimulationConfig
        {
            BallCount = BallCount,
            Gravity = 4.0,
            Restitution = 0.4,
            DeltaTime = 1.0 / 30.0
        };

        _flatSurface = GenerateFlatSurface();
        _hillySurface = GenerateHillySurface();
        _complexSurface = GenerateComplexSurface();

        _engine = new PhysicsEngine(config, _flatSurface, 1200, 600);
        _balls = SpawnBalls(BallCount);
    }

    private List<SurfacePoint> GenerateFlatSurface()
    {
        var surface = new List<SurfacePoint>();
        // Flat surface at y=400, from x=100 to x=1100
        for (double x = 100; x <= 1100; x += 3.0) // 3px spacing like real densification
        {
            surface.Add(new SurfacePoint(x, 400));
        }
        return surface;
    }

    private List<SurfacePoint> GenerateHillySurface()
    {
        var surface = new List<SurfacePoint>();
        for (double x = 100; x <= 1100; x += 3.0)
        {
            var y = 400 + 50 * Math.Sin(x * 0.02); // gentle sine waves
            surface.Add(new SurfacePoint(x, y));
        }
        return surface;
    }

    private List<SurfacePoint> GenerateComplexSurface()
    {
        var surface = new List<SurfacePoint>();
        // Complex terrain with multiple features
        for (double x = 100; x <= 1100; x += 3.0)
        {
            double y = 400;
            y += 50 * Math.Sin(x * 0.02);
            y += 25 * Math.Sin(x * 0.05);
            y += 10 * Math.Sin(x * 0.1);
            surface.Add(new SurfacePoint(x, y));
        }
        return surface;
    }

    private Ball[] SpawnBalls(int count)
    {
        var balls = new Ball[count];
        var random = new Random(42); // Fixed seed for reproducibility
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

    [Benchmark(Baseline = true, Description = "Flat Surface (y=400)")]
    public SimulationState BenchmarkFlatSurface()
    {
        // Clone balls to avoid state pollution
        var clonedBalls = CloneBalls(_balls!);
        return _engine!.Update(clonedBalls);
    }

    [Benchmark(Description = "Hilly Surface (sine waves)")]
    public SimulationState BenchmarkHillySurface()
    {
        var engine = new PhysicsEngine(new SimulationConfig
        {
            BallCount = BallCount,
            Gravity = 4.0,
            Restitution = 0.4,
            DeltaTime = 1.0 / 30.0
        }, _hillySurface!, 1200, 600);
        
        var clonedBalls = CloneBalls(_balls!);
        return engine.Update(clonedBalls);
    }

    [Benchmark(Description = "Complex Surface (multi-frequency)")]
    public SimulationState BenchmarkComplexSurface()
    {
        var engine = new PhysicsEngine(new SimulationConfig
        {
            BallCount = BallCount,
            Gravity = 4.0,
            Restitution = 0.4,
            DeltaTime = 1.0 / 30.0
        }, _complexSurface!, 1200, 600);
        
        var clonedBalls = CloneBalls(_balls!);
        return engine.Update(clonedBalls);
    }

    private Ball[] CloneBalls(Ball[] source)
    {
        var cloned = new Ball[source.Length];
        for (int i = 0; i < source.Length; i++)
        {
            cloned[i] = source[i];
        }
        return cloned;
    }
}
