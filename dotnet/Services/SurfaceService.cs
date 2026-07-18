using DotnetApi.Models;

namespace DotnetApi.Services;

public class SurfaceService
{
    private List<SurfacePoint> _surface = new();

    public List<SurfacePoint> Surface => _surface;

    public void SetSurface(List<SurfacePoint> points)
    {
        _surface = points ?? new();
    }

    public void AddPoint(double x, double y)
    {
        _surface.Add(new SurfacePoint(x, y));
    }

    public void Clear()
    {
        _surface.Clear();
    }

    public bool IsValid()
    {
        return _surface.Count >= 2;
    }

    public List<SurfacePoint> GetDefaultSurface()
    {
        // Right-angle corner: 45° up-left and 45° up-right meeting at bottom center
        var centerX = 400; // Will be adjusted based on canvas width
        var bottomY = 500; // Will be adjusted based on canvas height

        return new List<SurfacePoint>
        {
            new SurfacePoint(centerX - 300, bottomY),
            new SurfacePoint(centerX, bottomY + 100),
            new SurfacePoint(centerX + 300, bottomY)
        };
    }
}
