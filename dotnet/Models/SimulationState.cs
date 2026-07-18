namespace DotnetApi.Models;

public class SimulationState
{
    public Ball[] Balls { get; set; } = Array.Empty<Ball>();
    public long Timestamp { get; set; }
}
