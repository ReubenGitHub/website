using Microsoft.AspNetCore.SignalR;
using DotnetApi.Models;
using DotnetApi.Services;

namespace DotnetApi.Hubs;

public class PhysicsHub : Hub
{
    private readonly ILogger<PhysicsHub> _logger;
    private readonly ILogger<SimulationSession> _sessionLogger;
    private readonly SimulationStreamService _streamService;

    public PhysicsHub(
        ILogger<PhysicsHub> logger,
        ILogger<SimulationSession> sessionLogger,
        SimulationStreamService streamService)
    {
        _logger = logger;
        _sessionLogger = sessionLogger;
        _streamService = streamService;
    }
    
    /// <summary>
    /// Gets or creates a per-connection SimulationSession.
    /// Each SignalR connection gets its own independent session with isolated state.
    /// </summary>
    private SimulationSession GetOrCreateSession()
    {
        var key = $"session_{Context.ConnectionId}";
        if (Context.Items[key] is SimulationSession session)
        {
            return session;
        }
        
        // Create a new independent session for this connection
        var surfaceService = new SurfaceService();
        var newSession = new SimulationSession(surfaceService, _sessionLogger);
        Context.Items[key] = newSession;
        
        _logger.LogInformation("Created new session for client {ClientId}, session={SessionId}",
            Context.ConnectionId, System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(newSession));
        return newSession;
    }
    
    /// <summary>
    /// Disposes the session for a connection (called on disconnect).
    /// </summary>
    private void DisposeSessionForConnection()
    {
        var key = $"session_{Context.ConnectionId}";
        if (Context.Items[key] is SimulationSession session)
        {
            _logger.LogInformation("Disposing session for client {ClientId}, session={SessionId}, wasRunning={IsRunning}",
                Context.ConnectionId, System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(session), session.IsRunning);
            session.Dispose();
            Context.Items[key] = null!;
        }
    }

    public async Task StartSimulation(SimulationConfig config, List<SurfacePoint> surface)
    {
        var session = GetOrCreateSession();
        session.SetConfig(config);
        session.SetSurface(surface);

        _logger.LogInformation("StartSimulation called for client {ClientId}, session={SessionId}, isRunning={IsRunning}",
            Context.ConnectionId, System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(session), session.IsRunning);

        if (session.IsRunning)
        {
            // Simulation already running (e.g., after reconnect) - just restart streaming
            _logger.LogInformation("Simulation already running, restarting streaming for client {ClientId}", Context.ConnectionId);
            _streamService.StartStreaming(Context.ConnectionId, session);
            var reconnectState = session.GetState();
            await Clients.Caller.SendAsync("SimulationStarted", reconnectState);
            return;
        }

        // Session is not running (page refresh/new connection) - reset to start fresh
        _logger.LogInformation("Session not running, resetting for new simulation on client {ClientId}", Context.ConnectionId);
        session.Reset();
        session.Start();

        // Start streaming using background service (avoids ObjectDisposedException)
        _streamService.StartStreaming(Context.ConnectionId, session);

        // Notify frontend that simulation has started
        var state = session.GetState();
        await Clients.Caller.SendAsync("SimulationStarted", state);
    }

    public async Task PauseSimulation()
    {
        var session = GetOrCreateSession();
        session.Pause();
        // Send pause signal BEFORE stopping streaming (stopping may abort connection)
        try
        {
            await Clients.Caller.SendAsync("SimulationPaused");
            _logger.LogInformation("Sent SimulationPaused to {ConnectionId}", Context.ConnectionId);
        }
        catch
        {
            // Client may have disconnected, ignore
        }
        _streamService.StopStreaming(Context.ConnectionId);
        await Task.CompletedTask;
    }

    public async Task ResumeSimulation(SimulationConfig config, List<SurfacePoint> surface)
    {
        var session = GetOrCreateSession();
        session.SetConfig(config);
        session.SetSurface(surface);
        session.Resume();
        _streamService.StartStreaming(Context.ConnectionId, session);
        await Clients.Caller.SendAsync("SimulationResumed");
    }

    public async Task ResetSimulation()
    {
        var session = GetOrCreateSession();
        session.Reset();
        _streamService.StopStreaming(Context.ConnectionId);
        var state = session.GetState();
        await Clients.Caller.SendAsync("SimulationReset", state);
    }

    public async Task UpdateSurface(List<SurfacePoint> surface)
    {
        var session = GetOrCreateSession();
        session.SetSurface(surface);
        await Clients.Caller.SendAsync("SurfaceUpdated");
    }

    public async Task UpdateRestitution(double restitution)
    {
        var session = GetOrCreateSession();
        session.SetRestitution(restitution);
        await Task.CompletedTask;
    }

    public override async Task OnConnectedAsync()
    {
        await base.OnConnectedAsync();
        _logger.LogInformation("Client connected: {ClientId}", Context.ConnectionId);
    }

    public override async Task OnDisconnectedAsync(Exception? exception)
    {
        _logger.LogInformation("Client disconnected: {ClientId}, exception={Exception}",
            Context.ConnectionId, exception?.Message);
        // Stop streaming and dispose session for this client
        _streamService.StopStreaming(Context.ConnectionId);
        DisposeSessionForConnection();
        await base.OnDisconnectedAsync(exception);
    }
}
