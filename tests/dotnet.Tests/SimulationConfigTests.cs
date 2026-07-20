using DotnetApi.Models;

namespace dotnet.Tests;

public class SimulationConfigTests
{
    [Fact]
    public void DefaultBallCount_Is2000()
    {
        Assert.Equal(2000, new SimulationConfig().BallCount);
    }

    [Fact]
    public void DefaultGravity_Is9_8()
    {
        Assert.Equal(4.0, new SimulationConfig().Gravity);
    }

    [Fact]
    public void DefaultRestitution_Is0_7()
    {
        Assert.Equal(0.4, new SimulationConfig().Restitution);
    }

    [Fact]
    public void DefaultAirResistance_Is0_01()
    {
        Assert.Equal(0.03, new SimulationConfig().AirResistance);
    }

    [Fact]
    public void DefaultDeltaTime_Is1_30()
    {
        Assert.Equal(1.0 / 30.0, new SimulationConfig().DeltaTime);
    }

    [Fact]
    public void All_Properties_Are_Settable()
    {
        var config = new SimulationConfig
        {
            BallCount = 5000,
            Gravity = 15.0,
            Restitution = 0.9,
            AirResistance = 0.05,
            DeltaTime = 1.0 / 60.0
        };
        Assert.Equal(5000, config.BallCount);
        Assert.Equal(15.0, config.Gravity);
        Assert.Equal(0.9, config.Restitution);
        Assert.Equal(0.05, config.AirResistance);
        Assert.Equal(1.0 / 60.0, config.DeltaTime);
    }

    [Fact]
    public void Config_Properties_AreIndependent()
    {
        var config1 = new SimulationConfig { BallCount = 1000 };
        var config2 = new SimulationConfig { BallCount = 5000 };
        Assert.NotEqual(config1.BallCount, config2.BallCount);
        Assert.Equal(4.0, config1.Gravity);
        Assert.Equal(4.0, config2.Gravity);
    }
}
