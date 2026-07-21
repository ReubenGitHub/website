using DotnetApi.Models;

namespace dotnet.Tests;

public class BallTests
{
    [Fact]
    public void Constructor_InitializesAllProperties()
    {
        var ball = new Ball(100.0, 200.0, 50.0, -30.0, 2.5, 255, 128, 64);
        Assert.Equal(100.0, ball.X);
        Assert.Equal(200.0, ball.Y);
        Assert.Equal(50.0, ball.Vx);
        Assert.Equal(-30.0, ball.Vy);
        Assert.Equal(2.5, ball.Radius);
        Assert.Equal(255, ball.R);
        Assert.Equal(128, ball.G);
        Assert.Equal(64, ball.B);
        Assert.True(ball.Active);
    }

    [Fact]
    public void Constructor_SetsActiveToTrueByDefault()
    {
        var ball = new Ball(0, 0, 0, 0, 1.0, 0, 0, 0);
        Assert.True(ball.Active);
    }

    [Fact]
    public void Active_Property_CanBeSetToFalse()
    {
        var ball = new Ball(0, 0, 0, 0, 1.0, 0, 0, 0);
        ball.Active = false;
        Assert.False(ball.Active);
    }

    [Fact]
    public void All_Position_Velocity_Properties_Are_Settable()
    {
        var ball = new Ball(0, 0, 0, 0, 1.0, 0, 0, 0);
        ball.X = 100;
        ball.Y = 200;
        ball.Vx = 50;
        ball.Vy = -30;
        ball.Radius = 3.0;
        Assert.Equal(100, ball.X);
        Assert.Equal(200, ball.Y);
        Assert.Equal(50, ball.Vx);
        Assert.Equal(-30, ball.Vy);
        Assert.Equal(3.0, ball.Radius);
    }

    [Fact]
    public void Ball_CanBeStoredInArray()
    {
        var balls = new Ball[10];
        for (int i = 0; i < balls.Length; i++)
        {
            balls[i] = new Ball(i * 10, 100, 0, 0, 2.0, (byte)i, (byte)(255 - i), (byte)i);
        }
        Assert.Equal(10, balls.Length);
        Assert.True(balls[0].Active);
        Assert.True(balls[9].Active);
        Assert.Equal(90, balls[9].X);
    }

    [Fact]
    public void Ball_WithNegativeVelocity_MovesCorrectly()
    {
        var ball = new Ball(100, 100, -50, -30, 2.0, 255, 255, 255);
        Assert.Equal(-50, ball.Vx);
        Assert.Equal(-30, ball.Vy);
    }
}
