using Microsoft.AspNetCore.SignalR;
using DotnetApi.Models;
using DotnetApi.Services;

namespace DotnetApi.Hubs;

public class PhysicsHub : Hub
{
    private readonly IServiceProvider _services;
    private readonly ILogger<PhysicsHub> _logger;
    private readonly SimulationStreamService _streamService;
    private SimulationSession? _cachedSession;

    public PhysicsHub(IServiceProvider services, ILogger<PhysicsHub> logger, SimulationStreamService streamService)
    {
        _services = services;
        _logger = logger;
        _streamService = streamService;
    }
    
    private SimulationSession GetOrCreateSession()
    {
        // SimulationSession is registered as singleton — same instance for all connections
        var session = _services.GetRequiredService<SimulationSession>();
        return session;
    }
    
    private (SimulationSession Session, IDisposable Scope)? GetSessionWithScope()
    {
        var key = $"session_{Context.ConnectionId}";
        if (Context.Items[key] is Tuple<SimulationSession, IDisposable> tuple)
        {
            return (tuple.Item1, tuple.Item2);
        }
        return null;
    }

    public async Task StartSimulation(SimulationConfig config, List<SurfacePoint> surface)
    {
        var session = GetOrCreateSession();
        session.SetConfig(config);
        session.SetSurface(surface);
        _cachedSession = session;

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
        _cachedSession = null;
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
        var session = GetOrCreateSession();
        _logger.LogInformation("Client disconnected: {ClientId}, session={SessionId}, isRunning={IsRunning}, exception={Exception}",
            Context.ConnectionId, System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(session), session.IsRunning, exception?.Message);
        // Stop streaming for this client, but DO NOT dispose the session
        // Session persists across disconnects (singleton lifetime)
        _streamService.StopStreaming(Context.ConnectionId);
        await base.OnDisconnectedAsync(exception);
    }
}
