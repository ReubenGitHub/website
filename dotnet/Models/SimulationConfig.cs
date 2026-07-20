namespace DotnetApi.Models;

public class SimulationConfig
{
    public int BallCount { get; set; } = 2000;
    public double Gravity { get; set; } = 4.0;
    public double Restitution { get; set; } = 0.4;
    public double AirResistance { get; set; } = 0.03;
    public double DeltaTime { get; set; } = 1.0 / 30.0;
}
