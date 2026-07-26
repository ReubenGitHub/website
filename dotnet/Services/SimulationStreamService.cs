using Microsoft.AspNetCore.SignalR;
using System.Collections.Concurrent;
using DotnetApi.Hubs;

namespace DotnetApi.Services;

/// <summary>
/// Background service that manages simulation streaming independently of hub lifetime.
/// Solves the ObjectDisposedException when Task.Run tries to access Clients after hub is disposed.
/// </summary>
public class SimulationStreamService : BackgroundService
{
    private readonly IServiceProvider _services;
    private readonly ILogger<SimulationStreamService> _logger;
    private readonly IHubContext<PhysicsHub> _hubContext;
    private readonly ConcurrentDictionary<string, CancellationTokenSource> _activeStreams = new();

    public SimulationStreamService(
        IServiceProvider services,
        ILogger<SimulationStreamService> logger,
        IHubContext<PhysicsHub> hubContext)
    {
        _services = services;
        _logger = logger;
        _hubContext = hubContext;
    }

    public bool StartStreaming(string connectionId, SimulationSession session)
    {
        if (_activeStreams.TryGetValue(connectionId, out var existing))
        {
            _logger.LogInformation("Streaming already active for {ConnectionId}, cancelling...", connectionId);
            existing.Cancel();
            existing.Dispose();
            _activeStreams.TryRemove(connectionId, out _);
        }

        var cts = new CancellationTokenSource();
        if (_activeStreams.TryAdd(connectionId, cts))
        {
            _ = Task.Run(() => StreamingLoop(connectionId, session, cts.Token));
#if DEBUG
            _logger.LogInformation("Streaming started for {ConnectionId}", connectionId);
#else
            _logger.LogDebug("Streaming started for {ConnectionId}", connectionId);
#endif
            return true;
        }
        return false;
    }

    public void StopStreaming(string connectionId)
    {
        if (_activeStreams.TryRemove(connectionId, out var cts))
        {
            cts.Cancel();
            cts.Dispose();
#if DEBUG
            _logger.LogInformation("Streaming stopped for {ConnectionId}", connectionId);
#else
            _logger.LogDebug("Streaming stopped for {ConnectionId}", connectionId);
#endif
        }
    }

    private async Task StreamingLoop(string connectionId, SimulationSession session, CancellationToken ct)
    {
        var tickCount = 0;
        try
        {
#if DEBUG
            _logger.LogInformation("Streaming loop started for {ConnectionId}, session={SessionId}",
                connectionId, System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(session));
#else
            _logger.LogDebug("Streaming loop started for {ConnectionId}, session={SessionId}",
                connectionId, System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(session));
#endif

            while (!ct.IsCancellationRequested)
            {
                // Check cancellation before each iteration (prevents TaskCanceledException from Task.Delay)
                ct.ThrowIfCancellationRequested();
                
                var state = session.GetState();
                try
                {
                    await _hubContext.Clients.Client(connectionId).SendAsync("StateUpdate", state, ct);
                }
                catch (OperationCanceledException)
                {
                    // Cancellation during send is expected during stop
                    break;
                }
                catch (Exception ex)
                {
                    // Send failed (client disconnected/connection aborted), stop streaming
                    _logger.LogInformation("Send failed for {ConnectionId}: {Error}, stopping streaming", connectionId, ex.Message);
                    break;
                }

                // Only log tick progress in Development, not in Production
#if DEBUG
                if (++tickCount % 10 == 0)
                {
                    var activeCount = state.Balls.Count(b => b.Active);
                    _logger.LogInformation("Tick {Tick} for {ConnectionId}: {Active} active balls (running={IsRunning})",
                        tickCount, connectionId, activeCount, session.IsRunning);
                }
#else
                ++tickCount;
#endif

                // Check cancellation before delaying
                if (ct.IsCancellationRequested)
                {
                    break;
                }

                await Task.Delay((int)(session.Config.DeltaTime * 1000), ct);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in streaming loop for {ConnectionId}", connectionId);
        }
        finally
        {
            _activeStreams.TryRemove(connectionId, out _);
            // Note: SimulationPaused signal is sent by PauseSimulation hub method before stopping streaming.
            // We don't send it here to avoid sending after connection abort.
#if DEBUG
            _logger.LogInformation("Streaming loop ended for {ConnectionId}", connectionId);
#else
            _logger.LogDebug("Streaming loop ended for {ConnectionId}", connectionId);
#endif
        }
    }

    protected override Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("SimulationStreamService running at: {Time}", DateTimeOffset.Now);
        return Task.CompletedTask;
    }

    public override async Task StopAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("SimulationStreamService stopping...");
        foreach (var kvp in _activeStreams)
        {
            kvp.Value.Cancel();
        }
        await base.StopAsync(stoppingToken);
    }
}
