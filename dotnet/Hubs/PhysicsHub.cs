using Microsoft.AspNetCore.SignalR;
using DotnetApi.Models;
using DotnetApi.Services;

namespace DotnetApi.Hubs;

public class PhysicsHub : Hub
{
    private readonly SimulationSession _session;
    private readonly ILogger<PhysicsHub> _logger;

    public PhysicsHub(SimulationSession session, ILogger<PhysicsHub> logger)
    {
        _session = session;
        _logger = logger;
    }

    public async Task StartSimulation(SimulationConfig config, List<SurfacePoint> surface)
    {
        _session.SetConfig(config);
        _session.SetSurface(surface);
        _session.Start();

        _logger.LogInformation("Simulation started for client {ClientId} with {BallCount} balls", 
            Context.ConnectionId, config.BallCount);

        // Stream positions
        Task.Run(async () =>
        {
            while (_session.IsRunning)
            {
                var state = _session.GetState();
                await Clients.Caller.SendAsync("StateUpdate", state);
                await Task.Delay((int)(_session.Config.DeltaTime * 1000));
            }
        });
    }

    public async Task PauseSimulation()
    {
        _session.Pause();
        await Clients.Caller.SendAsync("SimulationPaused");
    }

    public async Task ResumeSimulation()
    {
        _session.Resume();
        await Clients.Caller.SendAsync("SimulationResumed");
    }

    public async Task ResetSimulation()
    {
        _session.Reset();
        var state = _session.GetState();
        await Clients.Caller.SendAsync("SimulationReset", state);
    }

    public async Task UpdateSurface(List<SurfacePoint> surface)
    {
        _session.SetSurface(surface);
        await Clients.Caller.SendAsync("SurfaceUpdated");
    }

    public override async Task OnConnectedAsync()
    {
        await base.OnConnectedAsync();
        _logger.LogInformation("Client connected: {ClientId}", Context.ConnectionId);
    }

    public override async Task OnDisconnectedAsync(Exception? exception)
    {
        _session.Stop();
        await base.OnDisconnectedAsync(exception);
        _logger.LogInformation("Client disconnected: {ClientId}", Context.ConnectionId);
    }
}
