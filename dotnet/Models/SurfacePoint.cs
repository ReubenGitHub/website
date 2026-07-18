namespace DotnetApi.Models;

public class SurfacePoint
{
    public double X { get; set; }
    public double Y { get; set; }

    public SurfacePoint() { }

    public SurfacePoint(double x, double y)
    {
        X = x;
        Y = y;
    }
}
