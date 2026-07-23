using DotnetApi.Models;
using DotnetApi.Services;
using Microsoft.Extensions.Logging.Abstractions;

namespace dotnet.Tests;

public class SimulationSessionTests
{
    private readonly SurfaceService _surfaceService;
    private readonly SimulationSession _session;

    public SimulationSessionTests()
    {
        _surfaceService = new SurfaceService();
        _session = new SimulationSession(_surfaceService, NullLogger<SimulationSession>.Instance);
    }

    [Fact]
    public void Constructor_InitializesWithDefaultConfig()
    {
        Assert.Equal(2000, _session.Config.BallCount);
        Assert.Equal(4.0, _session.Config.Gravity);
        Assert.Equal(0.4, _session.Config.Restitution);
        Assert.Equal(1.0 / 30.0, _session.Config.DeltaTime);
    }

    [Fact]
    public void Constructor_InitializesIsRunningAsFalse()
    {
        Assert.False(_session.IsRunning);
    }

    [Fact]
    public void Constructor_InitializesWithEmptyBalls()
    {
        var state = _session.GetState();
        Assert.Empty(state.Balls);
    }

    [Fact]
    public void SetConfig_UpdatesConfigValues()
    {
        var config = new SimulationConfig
        {
            BallCount = 5000,
            Gravity = 15.0,
            Restitution = 0.9,
            DeltaTime = 1.0 / 60.0
        };
        _session.SetConfig(config);
        Assert.Equal(5000, _session.Config.BallCount);
        Assert.Equal(15.0, _session.Config.Gravity);
        Assert.Equal(0.9, _session.Config.Restitution);
        Assert.Equal(1.0 / 60.0, _session.Config.DeltaTime);
    }

    [Fact]
    public void SetSurface_ForwardsToSurfaceService()
    {
        var points = new List<SurfacePoint>
        {
            new SurfacePoint(0, 0),
            new SurfacePoint(2, 2),   // Close points to avoid densification
            new SurfacePoint(4, 0)
        };
        _session.SetSurface(points);
        Assert.Equal(3, _surfaceService.Surface.Count);
        Assert.True(_surfaceService.IsValid());
    }

    [Fact]
    public void Start_SetsIsRunningToTrue()
    {
        _session.Start();
        Assert.True(_session.IsRunning);
    }

    [Fact]
    public void Start_SpawnsBalls()
    {
        _session.SetConfig(new SimulationConfig { BallCount = 100 });
        _session.SetSurface(new List<SurfacePoint>
        {
            new SurfacePoint(100, 400),
            new SurfacePoint(200, 500),
            new SurfacePoint(300, 400)
        });
        _session.Start();
        var state = _session.GetState();
        Assert.NotEmpty(state.Balls);
        Assert.Equal(100, state.Balls.Length);
    }

    [Fact]
    public void Start_BallsAreActive()
    {
        _session.SetConfig(new SimulationConfig { BallCount = 50 });
        _session.SetSurface(new List<SurfacePoint>
        {
            new SurfacePoint(100, 400),
            new SurfacePoint(200, 500),
            new SurfacePoint(300, 400)
        });
        _session.Start();
        var state = _session.GetState();
        var activeBalls = state.Balls.Count(b => b.Active);
        Assert.Equal(50, activeBalls);
    }

    [Fact]
    public void Start_BallsHaveValidColors()
    {
        _session.SetConfig(new SimulationConfig { BallCount = 10 });
        _session.SetSurface(new List<SurfacePoint>
        {
            new SurfacePoint(100, 400),
            new SurfacePoint(300, 400)
        });
        _session.Start();
        var state = _session.GetState();
        foreach (var ball in state.Balls)
        {
            Assert.InRange(ball.R, 0, 255);
            Assert.InRange(ball.G, 0, 255);
            Assert.InRange(ball.B, 0, 255);
        }
    }

    [Fact]
    public void Pause_SetsIsRunningToFalse()
    {
        _session.Start();
        Assert.True(_session.IsRunning);
        _session.Pause();
        Assert.False(_session.IsRunning);
    }

    [Fact]
    public void Resume_SetsIsRunningToTrue()
    {
        _session.Start();
        _session.Pause();
        Assert.False(_session.IsRunning);
        _session.Resume();
        Assert.True(_session.IsRunning);
    }

    [Fact]
    public void Reset_SetsIsRunningToFalse()
    {
        _session.Start();
        Assert.True(_session.IsRunning);
        _session.Reset();
        Assert.False(_session.IsRunning);
    }

    [Fact]
    public void Reset_ClearsBalls()
    {
        _session.SetConfig(new SimulationConfig { BallCount = 100 });
        _session.SetSurface(new List<SurfacePoint>
        {
            new SurfacePoint(100, 400),
            new SurfacePoint(300, 400)
        });
        _session.Start();
        Assert.NotEmpty(_session.GetState().Balls);
        _session.Reset();
        var state = _session.GetState();
        Assert.Empty(state.Balls);
    }

    [Fact]
    public void Reset_AllowsRestart()
    {
        _session.SetConfig(new SimulationConfig { BallCount = 50 });
        _session.SetSurface(new List<SurfacePoint>
        {
            new SurfacePoint(100, 400),
            new SurfacePoint(300, 400)
        });
        _session.Start();
        _session.Reset();
        _session.Start();
        Assert.True(_session.IsRunning);
        var state = _session.GetState();
        Assert.NotEmpty(state.Balls);
        Assert.Equal(50, state.Balls.Length);
    }

    [Fact]
    public void GetState_ReturnsSimulationState()
    {
        _session.SetConfig(new SimulationConfig { BallCount = 10 });
        _session.SetSurface(new List<SurfacePoint>
        {
            new SurfacePoint(100, 400),
            new SurfacePoint(300, 400)
        });
        var state = _session.GetState();
        Assert.NotNull(state);
        Assert.IsType<Ball[]>(state.Balls);
        Assert.NotEqual(0L, state.Timestamp);
    }

    [Fact]
    public void GetState_ReturnsEmptyBalls_WhenNotStarted()
    {
        var state = _session.GetState();
        Assert.Empty(state.Balls);
    }

    [Fact]
    public void FullLifecycle_StartPauseResumePreservesBalls()
    {
        _session.SetConfig(new SimulationConfig { BallCount = 100, Gravity = 9.8 });
        _session.SetSurface(new List<SurfacePoint>
        {
            new SurfacePoint(100, 400),
            new SurfacePoint(200, 500),
            new SurfacePoint(300, 400)
        });
        _session.Start();
        Thread.Sleep(200);
        var stateBeforePause = _session.GetState();
        var ballCountBefore = stateBeforePause.Balls.Length;
        _session.Pause();
        var statePaused = _session.GetState();
        _session.Resume();
        Thread.Sleep(100);
        var stateAfterResume = _session.GetState();
        Assert.Equal(ballCountBefore, statePaused.Balls.Length);
        Assert.Equal(ballCountBefore, stateAfterResume.Balls.Length);
    }

    [Fact]
    public void MultipleStartCalls_DontThrow()
    {
        _session.SetConfig(new SimulationConfig { BallCount = 10 });
        _session.SetSurface(new List<SurfacePoint>
        {
            new SurfacePoint(100, 400),
            new SurfacePoint(300, 400)
        });
        _session.Start();
        _session.Start();
        _session.Start();
        Assert.True(_session.IsRunning);
    }

    [Fact]
    public void MultiplePauseResumeCycles_DontThrow()
    {
        _session.SetConfig(new SimulationConfig { BallCount = 10 });
        _session.SetSurface(new List<SurfacePoint>
        {
            new SurfacePoint(100, 400),
            new SurfacePoint(300, 400)
        });
        _session.Start();
        for (int i = 0; i < 5; i++)
        {
            _session.Pause();
            Assert.False(_session.IsRunning);
            _session.Resume();
            Assert.True(_session.IsRunning);
            Thread.Sleep(50);
        }
    }

    [Fact]
    public void Dispose_CleansUpResources()
    {
        _session.SetConfig(new SimulationConfig { BallCount = 10 });
        _session.SetSurface(new List<SurfacePoint>
        {
            new SurfacePoint(100, 400),
            new SurfacePoint(300, 400)
        });
        _session.Start();
        _session.Dispose();
        Assert.False(_session.IsRunning);
    }
}
