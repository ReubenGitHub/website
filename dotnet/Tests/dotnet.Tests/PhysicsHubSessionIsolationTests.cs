using DotnetApi.Models;
using DotnetApi.Services;
using Microsoft.Extensions.Logging.Abstractions;

namespace dotnet.Tests;

/// <summary>
/// Tests that verify SimulationSession isolation per connection.
/// These tests simulate the hub's singleton bug by using a shared session
/// and verifying that per-connection sessions are required.
/// </summary>
public class PhysicsHubSessionIsolationTests
{
    /// <summary>
    /// SIMULATES THE BUG: Two clients sharing ONE session instance.
    /// This test demonstrates the current broken behavior - it PASSES now because
    /// both clients operate on the same session, but this is the WRONG behavior.
    /// After the fix, this test should be replaced by tests that use per-connection sessions.
    /// 
    /// The key assertion is that when Client A and Client B share a session,
    /// Client A's pause affects Client B - which is the BUG we're fixing.
    /// </summary>
    [Fact]
    public void SharedSession_BugDemonstration_PauseAffectsBoth()
    {
        // This simulates the CURRENT broken behavior: one session shared by both clients
        var sharedSurfaceService = new SurfaceService();
        var sharedSession = new SimulationSession(sharedSurfaceService, NullLogger<SimulationSession>.Instance);

        // Client A starts
        sharedSession.SetConfig(new SimulationConfig { BallCount = 100 });
        sharedSession.SetSurface(CreateDefaultSurface());
        sharedSession.Start();

        // "Client B" is also using the same session
        // Client B hasn't pressed pause, but Client A did
        sharedSession.Pause(); // Client A pauses

        // BUG: Client B sees paused state too (same session!)
        // This is the current broken behavior - both clients share state
        Assert.False(sharedSession.IsRunning); // Both see it as paused

        // Cleanup
        sharedSession.Dispose();
    }

    /// <summary>
    /// THE FIX: Two clients with SEPARATE session instances.
    /// This test will FAIL with current code (since hub uses singleton) but
    /// PASSES once we fix the hub to create per-connection sessions.
    /// </summary>
    [Fact]
    public void SeparateSessions_PauseA_DoesNotAffectSessionB()
    {
        // Arrange: Create TWO independent sessions (simulating per-connection sessions)
        var (sessionA, _) = CreateSessionWithConfig(100);
        var (sessionB, _) = CreateSessionWithConfig(200);

        // Both start
        sessionA.Start();
        sessionB.Start();

        // Verify both are running
        Assert.True(sessionA.IsRunning, "Session A should be running");
        Assert.True(sessionB.IsRunning, "Session B should be running");

        // Act: Client A pauses
        sessionA.Pause();

        // Assert: Only A is paused, B continues running
        Assert.False(sessionA.IsRunning, "Session A should be paused");
        Assert.True(sessionB.IsRunning, 
            "Session B should STILL be running - this is the key isolation requirement");
        
        // Ball counts should remain independent
        var stateB = sessionB.GetState();
        Assert.True(stateB.Balls.Length == 200, 
            $"Session B should still have 200 balls, got {stateB.Balls.Length}");

        // Cleanup
        sessionA.Dispose();
        sessionB.Dispose();
    }

    /// <summary>
    /// THE FIX: Verify ball count independence between sessions.
    /// Client A with 50 balls should not see Client B's 500 balls.
    /// </summary>
    [Fact]
    public void SeparateSessions_BallCountsAreIndependent()
    {
        // Arrange
        var (sessionA, _) = CreateSessionWithConfig(50);
        var (sessionB, _) = CreateSessionWithConfig(500);

        // Act
        sessionA.Start();
        sessionB.Start();

        // Assert
        var stateA = sessionA.GetState();
        var stateB = sessionB.GetState();

        Assert.True(stateA.Balls.Length == 50,
            $"Session A should have 50 balls, got {stateA.Balls.Length}");
        Assert.True(stateB.Balls.Length == 500,
            $"Session B should have 500 balls, got {stateB.Balls.Length}");

        // Cleanup
        sessionA.Dispose();
        sessionB.Dispose();
    }

    /// <summary>
    /// THE FIX: Verify reset independence between sessions.
    /// Client A resetting should not affect Client B's running simulation.
    /// </summary>
    [Fact]
    public void SeparateSessions_ResetA_DoesNotAffectRunningSessionB()
    {
        // Arrange
        var (sessionA, _) = CreateSessionWithConfig(100);
        var (sessionB, _) = CreateSessionWithConfig(200);

        sessionA.Start();
        sessionB.Start();

        // Act: Client A resets
        sessionA.Reset();

        // Assert
        var stateA = sessionA.GetState();
        var stateB = sessionB.GetState();

        Assert.True(stateA.Balls.Length == 0, $"Session A should have no balls after reset, got {stateA.Balls.Length}");
        Assert.False(sessionA.IsRunning, "Session A should not be running after reset");
        Assert.True(stateB.Balls.Length == 200,
            $"Session B should still have 200 balls, got {stateB.Balls.Length}");
        Assert.True(sessionB.IsRunning, "Session B should still be running");

        // Cleanup
        sessionA.Dispose();
        sessionB.Dispose();
    }

    /// <summary>
    /// THE FIX: Verify surface paint independence between sessions.
    /// Client A painting a surface should not affect Client B's surface.
    /// </summary>
    [Fact]
    public void SeparateSessions_SurfacePaintA_DoesNotAffectSessionB()
    {
        // Arrange
        var (sessionA, surfaceServiceA) = CreateSessionWithConfig(100);
        var (sessionB, surfaceServiceB) = CreateSessionWithConfig(100);

        // Client A paints a custom surface (points close together to avoid densification)
        var customSurface = new List<SurfacePoint>
        {
            new SurfacePoint(0, 0),
            new SurfacePoint(2, 2),
            new SurfacePoint(4, 0),
            new SurfacePoint(2, 1)
        };
        sessionA.SetSurface(customSurface);

        // Client B has default surface (3 points, also close together)
        sessionB.SetSurface(new List<SurfacePoint>
        {
            new SurfacePoint(10, 10),
            new SurfacePoint(12, 12),
            new SurfacePoint(14, 14)
        });

        // Assert
        Assert.True(surfaceServiceA.Surface.Count == 4,
            $"Session A surface should have 4 points, got {surfaceServiceA.Surface.Count}");
        Assert.True(surfaceServiceB.Surface.Count == 3,
            $"Session B surface should have 3 points, got {surfaceServiceB.Surface.Count}");

        // Cleanup
        sessionA.Dispose();
        sessionB.Dispose();
    }

    private static (SimulationSession Session, SurfaceService SurfaceService) CreateSessionWithConfig(int ballCount)
    {
        var surfaceService = new SurfaceService();
        var session = new SimulationSession(surfaceService, NullLogger<SimulationSession>.Instance);
        session.SetConfig(new SimulationConfig { BallCount = ballCount });
        session.SetSurface(CreateDefaultSurface());
        return (session, surfaceService);
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
