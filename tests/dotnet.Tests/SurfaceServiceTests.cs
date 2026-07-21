using DotnetApi.Models;
using DotnetApi.Services;

namespace dotnet.Tests;

public class SurfaceServiceTests
{
    [Fact]
    public void SetSurface_WithValidPoints_SetsSurface()
    {
        var service = new SurfaceService();
        var points = new List<SurfacePoint>
        {
            new SurfacePoint(0, 0),
            new SurfacePoint(2, 2)  // Close points to avoid densification changing count
        };

        service.SetSurface(points);

        Assert.Equal(2, service.Surface.Count);
        Assert.Equal(0, service.Surface[0].X);
        Assert.Equal(2, service.Surface[1].X);
    }

    [Fact]
    public void SetSurface_WithNull_SetsEmptySurface()
    {
        var service = new SurfaceService();
        service.SetSurface(null!);
        Assert.Empty(service.Surface);
    }

    [Fact]
    public void SetSurface_OverwritesPreviousSurface()
    {
        var service = new SurfaceService();
        service.SetSurface(new List<SurfacePoint> { new SurfacePoint(0, 0) });
        service.SetSurface(new List<SurfacePoint>
        {
            new SurfacePoint(10, 10),
            new SurfacePoint(12, 12),  // Close points to avoid densification
            new SurfacePoint(14, 14)
        });
        Assert.Equal(3, service.Surface.Count);
    }

    [Fact]
    public void AddPoint_AddsPointToSurface()
    {
        var service = new SurfaceService();
        service.SetSurface(new List<SurfacePoint> { new SurfacePoint(0, 0) });
        service.AddPoint(100, 200);
        Assert.Equal(2, service.Surface.Count);
        Assert.Equal(100, service.Surface[1].X);
        Assert.Equal(200, service.Surface[1].Y);
    }

    [Fact]
    public void Clear_RemovesAllPoints()
    {
        var service = new SurfaceService();
        service.SetSurface(new List<SurfacePoint>
        {
            new SurfacePoint(0, 0),
            new SurfacePoint(100, 100),
            new SurfacePoint(200, 200)
        });
        service.Clear();
        Assert.Empty(service.Surface);
    }

    [Fact]
    public void IsValid_ReturnsFalse_WhenLessThan2Points()
    {
        var service = new SurfaceService();
        Assert.False(service.IsValid());
        service.SetSurface(new List<SurfacePoint> { new SurfacePoint(0, 0) });
        Assert.False(service.IsValid());
    }

    [Fact]
    public void IsValid_ReturnsTrue_When2OrMorePoints()
    {
        var service = new SurfaceService();
        service.SetSurface(new List<SurfacePoint>
        {
            new SurfacePoint(0, 0),
            new SurfacePoint(100, 100)
        });
        Assert.True(service.IsValid());
    }

    [Fact]
    public void GetDefaultSurface_Returns3Points()
    {
        var service = new SurfaceService();
        var defaultSurface = service.GetDefaultSurface();
        Assert.Equal(3, defaultSurface.Count);
        Assert.Equal(100, defaultSurface[0].X);
        Assert.Equal(500, defaultSurface[0].Y);
        Assert.Equal(400, defaultSurface[1].X);
        Assert.Equal(600, defaultSurface[1].Y);
        Assert.Equal(700, defaultSurface[2].X);
        Assert.Equal(500, defaultSurface[2].Y);
    }

    [Fact]
    public void Surface_Property_ReturnsSameInstance()
    {
        var service = new SurfaceService();
        var points = new List<SurfacePoint> { new SurfacePoint(0, 0) };
        service.SetSurface(points);
        Assert.Same(points, service.Surface);
    }
}
