using DotnetApi.Models;

namespace DotnetApi.Services;

public class SurfaceService
{
    private List<SurfacePoint> _surface = new();

    public List<SurfacePoint> Surface => _surface;

    public void SetSurface(List<SurfacePoint> points)
    {
        _surface = points ?? new();
        DensifySurface();
    }

    /// <summary>
    /// Inserts additional points between surface points that are too far apart
    /// to prevent balls from escaping through gaps in hand-drawn surfaces.
    /// </summary>
    private void DensifySurface()
    {
        if (_surface.Count < 2)
            return;

        const double maxGap = 3.0; // Maximum distance between adjacent points (pixels)
        var densified = new List<SurfacePoint>();

        for (int i = 0; i < _surface.Count - 1; i++)
        {
            densified.Add(_surface[i]);
            var p1 = _surface[i];
            var p2 = _surface[i + 1];

            double dx = p2.X - p1.X;
            double dy = p2.Y - p1.Y;
            double distance = Math.Sqrt(dx * dx + dy * dy);

            if (distance > maxGap)
            {
                int steps = (int)Math.Ceiling(distance / maxGap);
                for (int s = 1; s < steps; s++)
                {
                    double t = (double)s / steps;
                    densified.Add(new SurfacePoint(
                        p1.X + dx * t,
                        p1.Y + dy * t
                    ));
                }
            }
        }

        // Add the last point
        if (_surface.Count > 0)
            densified.Add(_surface[^1]);

        _surface = densified;
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
