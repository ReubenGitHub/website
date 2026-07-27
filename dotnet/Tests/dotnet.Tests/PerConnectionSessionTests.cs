using DotnetApi.Models;
using DotnetApi.Services;
using Microsoft.Extensions.Logging.Abstractions;

namespace dotnet.Tests;

/// <summary>
/// Tests that verify SimulationSession instances are properly isolated per connection.
/// Each client connection should have its own independent session state.
/// </summary>
public class PerConnectionSessionTests
{
    /// <summary>
    /// Verifies that two independent SimulationSession instances have completely separate state.
    /// This is the core requirement: Client A's session must not affect Client B's session.
    /// </summary>
    [Fact]
    public void TwoSessions_HaveIndependentRunningState()
    {
        var surfaceServiceA = CreateSurfaceService();
        var sessionA = new SimulationSession(surfaceServiceA, NullLogger<SimulationSession>.Instance);
        
        var surfaceServiceB = CreateSurfaceService();
        var sessionB = new SimulationSession(surfaceServiceB, NullLogger<SimulationSession>.Instance);

        // Client A starts simulation
        sessionA.SetConfig(new SimulationConfig { BallCount = 100 });
        sessionA.SetSurface(CreateDefaultSurface());
        sessionA.Start();

        // Client B has not started
        Assert.True(sessionA.IsRunning);
        Assert.False(sessionB.IsRunning);

        // Cleanup
        sessionA.Dispose();
    }

    /// <summary>
    /// Verifies that pausing one session does not affect another session's running state.
    /// This is the key bug: if Client A pauses, Client B should NOT see it paused.
    /// </summary>
    [Fact]
    public void PauseSessionA_DoesNotAffectSessionB()
    {
        var surfaceServiceA = CreateSurfaceService();
        var sessionA = new SimulationSession(surfaceServiceA, NullLogger<SimulationSession>.Instance);
        
        var surfaceServiceB = CreateSurfaceService();
        var sessionB = new SimulationSession(surfaceServiceB, NullLogger<SimulationSession>.Instance);

        // Both clients start simulations
        sessionA.SetConfig(new SimulationConfig { BallCount = 100 });
        sessionA.SetSurface(CreateDefaultSurface());
        sessionA.Start();

        sessionB.SetConfig(new SimulationConfig { BallCount = 200 });
        sessionB.SetSurface(CreateDefaultSurface());
        sessionB.Start();

        // Both should be running
        Assert.True(sessionA.IsRunning);
        Assert.True(sessionB.IsRunning);

        // Client A pauses
        sessionA.Pause();

        // Client A should be paused, Client B should still be running
        Assert.False(sessionA.IsRunning);
        Assert.True(sessionB.IsRunning, "Session B should still be running after Session A is paused");

        // Cleanup
        sessionA.Dispose();
        sessionB.Dispose();
    }

    /// <summary>
    /// Verifies that each session spawns its own configured number of balls.
    /// If Client A requests 50 balls and Client B requests 500 balls,
    /// each should see their own ball count.
    /// </summary>
    [Fact]
    public void Sessions_HaveIndependentBallCounts()
    {
        var surfaceServiceA = CreateSurfaceService();
        var sessionA = new SimulationSession(surfaceServiceA, NullLogger<SimulationSession>.Instance);
        
        var surfaceServiceB = CreateSurfaceService();
        var sessionB = new SimulationSession(surfaceServiceB, NullLogger<SimulationSession>.Instance);

        // Client A starts with 50 balls
        sessionA.SetConfig(new SimulationConfig { BallCount = 50 });
        sessionA.SetSurface(CreateDefaultSurface());
        sessionA.Start();

        // Client B starts with 500 balls
        sessionB.SetConfig(new SimulationConfig { BallCount = 500 });
        sessionB.SetSurface(CreateDefaultSurface());
        sessionB.Start();

        var stateA = sessionA.GetState();
        var stateB = sessionB.GetState();

        Assert.True(stateA.Balls.Length == 50, $"Session A should have 50 balls, got {stateA.Balls.Length}");
        Assert.True(stateB.Balls.Length == 500, $"Session B should have 500 balls, got {stateB.Balls.Length}");

        // Cleanup
        sessionA.Dispose();
        sessionB.Dispose();
    }

    /// <summary>
    /// Verifies that resetting one session does not affect another session's balls or running state.
    /// </summary>
    [Fact]
    public void ResetSessionA_DoesNotAffectSessionB()
    {
        var surfaceServiceA = CreateSurfaceService();
        var sessionA = new SimulationSession(surfaceServiceA, NullLogger<SimulationSession>.Instance);
        
        var surfaceServiceB = CreateSurfaceService();
        var sessionB = new SimulationSession(surfaceServiceB, NullLogger<SimulationSession>.Instance);

        // Both clients start simulations
        sessionA.SetConfig(new SimulationConfig { BallCount = 100 });
        sessionA.SetSurface(CreateDefaultSurface());
        sessionA.Start();

        sessionB.SetConfig(new SimulationConfig { BallCount = 200 });
        sessionB.SetSurface(CreateDefaultSurface());
        sessionB.Start();

        // Client A resets
        sessionA.Reset();

        // Client A should have no balls, Client B should still have its balls
        var stateA = sessionA.GetState();
        var stateB = sessionB.GetState();

        Assert.True(stateA.Balls.Length == 0, $"Session A should have no balls after reset, got {stateA.Balls.Length}");
        Assert.False(sessionA.IsRunning, "Session A should not be running after reset");
        Assert.True(stateB.Balls.Length == 200, $"Session B should still have 200 balls, got {stateB.Balls.Length}");
        Assert.True(sessionB.IsRunning, "Session B should still be running after Session A is reset");

        // Cleanup
        sessionA.Dispose();
        sessionB.Dispose();
    }

    /// <summary>
    /// Verifies that setting different configs on each session is independent.
    /// </summary>
    [Fact]
    public void Sessions_HaveIndependentConfigs()
    {
        var surfaceServiceA = CreateSurfaceService();
        var sessionA = new SimulationSession(surfaceServiceA, NullLogger<SimulationSession>.Instance);
        
        var surfaceServiceB = CreateSurfaceService();
        var sessionB = new SimulationSession(surfaceServiceB, NullLogger<SimulationSession>.Instance);

        sessionA.SetConfig(new SimulationConfig { BallCount = 100, Gravity = 9.8, Restitution = 0.5 });
        sessionB.SetConfig(new SimulationConfig { BallCount = 500, Gravity = 4.0, Restitution = 0.9 });

        Assert.Equal(9.8, sessionA.Config.Gravity);
        Assert.Equal(0.5, sessionA.Config.Restitution);
        Assert.Equal(4.0, sessionB.Config.Gravity);
        Assert.Equal(0.9, sessionB.Config.Restitution);

        // Cleanup
        sessionA.Dispose();
        sessionB.Dispose();
    }

    /// <summary>
    /// Verifies that surface painting on one session doesn't affect another.
    /// </summary>
    [Fact]
    public void UpdateSurfaceOnSessionA_DoesNotAffectSessionB()
    {
        var surfaceServiceA = CreateSurfaceService();
        var sessionA = new SimulationSession(surfaceServiceA, NullLogger<SimulationSession>.Instance);
        
        var surfaceServiceB = CreateSurfaceService();
        var sessionB = new SimulationSession(surfaceServiceB, NullLogger<SimulationSession>.Instance);

        // Client A paints a custom surface (points close together to avoid densification)
        var customSurface = new List<SurfacePoint>
        {
            new SurfacePoint(0, 0),
            new SurfacePoint(2, 2),
            new SurfacePoint(4, 0)
        };
        sessionA.SetSurface(customSurface);

        // Client B has default surface (3 points, also close together)
        sessionB.SetSurface(new List<SurfacePoint>
        {
            new SurfacePoint(10, 10),
            new SurfacePoint(12, 12),
            new SurfacePoint(14, 14)
        });

        // Both surfaces should have exactly 3 points (no densification with close points)
        Assert.True(surfaceServiceA.Surface.Count == 3, $"Session A surface should have 3 points, got {surfaceServiceA.Surface.Count}");
        Assert.True(surfaceServiceB.Surface.Count == 3, $"Session B surface should have 3 points, got {surfaceServiceB.Surface.Count}");

        // Verify the surfaces are different (different Y values)
        Assert.True(surfaceServiceA.Surface[0].Y != surfaceServiceB.Surface[0].Y,
            "Session A and B should have different surface Y values");

        // Cleanup
        sessionA.Dispose();
        sessionB.Dispose();
    }

    /// <summary>
    /// Verifies that Client B starting after Client A started does not share Client A's state.
    /// This simulates the real-world scenario where one client loads the page later.
    /// </summary>
    [Fact]
    public async Task SequentialSessionStarts_HaveIndependentState()
    {
        var surfaceServiceA = CreateSurfaceService();
        var sessionA = new SimulationSession(surfaceServiceA, NullLogger<SimulationSession>.Instance);
        
        var surfaceServiceB = CreateSurfaceService();
        var sessionB = new SimulationSession(surfaceServiceB, NullLogger<SimulationSession>.Instance);

        // Client A starts first with 100 balls
        sessionA.SetConfig(new SimulationConfig { BallCount = 100 });
        sessionA.SetSurface(CreateDefaultSurface());
        sessionA.Start();

        // Wait a bit to simulate Client B loading later
        await Task.Delay(100);

        // Client B starts with 300 balls - should NOT inherit Client A's state
        sessionB.SetConfig(new SimulationConfig { BallCount = 300 });
        sessionB.SetSurface(CreateDefaultSurface());
        sessionB.Start();

        var stateA = sessionA.GetState();
        var stateB = sessionB.GetState();

        Assert.True(stateA.Balls.Length == 100, $"Session A should have 100 balls, got {stateA.Balls.Length}");
        Assert.True(stateB.Balls.Length == 300, $"Session B should have 300 balls, got {stateB.Balls.Length}");

        // Cleanup
        sessionA.Dispose();
        sessionB.Dispose();
    }

    /// <summary>
    /// Verifies that Client A resuming does not affect Client B's paused state.
    /// </summary>
    [Fact]
    public void ResumeSessionA_DoesNotAffectPausedSessionB()
    {
        var surfaceServiceA = CreateSurfaceService();
        var sessionA = new SimulationSession(surfaceServiceA, NullLogger<SimulationSession>.Instance);
        
        var surfaceServiceB = CreateSurfaceService();
        var sessionB = new SimulationSession(surfaceServiceB, NullLogger<SimulationSession>.Instance);

        // Both start
        sessionA.SetConfig(new SimulationConfig { BallCount = 100 });
        sessionA.SetSurface(CreateDefaultSurface());
        sessionA.Start();

        sessionB.SetConfig(new SimulationConfig { BallCount = 200 });
        sessionB.SetSurface(CreateDefaultSurface());
        sessionB.Start();

        // Client B pauses
        sessionB.Pause();
        Assert.False(sessionB.IsRunning);
        Assert.True(sessionA.IsRunning);

        // Client A pauses too
        sessionA.Pause();
        Assert.False(sessionA.IsRunning);
        Assert.False(sessionB.IsRunning);

        // Client A resumes
        sessionA.Resume();
        Assert.True(sessionA.IsRunning);
        Assert.False(sessionB.IsRunning, "Session B should still be paused after Session A resumes");

        // Cleanup
        sessionA.Dispose();
        sessionB.Dispose();
    }

    private static SurfaceService CreateSurfaceService()
    {
        return new SurfaceService();
    }

    private static List<SurfacePoint> CreateDefaultSurface()
    {
        return new List<SurfacePoint>
        {
            new SurfacePoint(100, 400),
            new SurfacePoint(200, 500),
            new SurfacePoint(300, 400)
        };
    }
}
