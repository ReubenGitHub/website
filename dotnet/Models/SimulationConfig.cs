namespace DotnetApi.Models;

public class SimulationConfig
{
    public int BallCount { get; set; } = 2000;
    public double Gravity { get; set; } = 9.8;
    public double Restitution { get; set; } = 0.7;
    public double AirResistance { get; set; } = 0.01;
    public double DeltaTime { get; set; } = 1.0 / 30.0;
}
