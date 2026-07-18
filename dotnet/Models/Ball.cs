namespace DotnetApi.Models;

public struct Ball
{
    public double X { get; set; }
    public double Y { get; set; }
    public double Vx { get; set; }
    public double Vy { get; set; }
    public double Radius { get; set; }
    public byte R { get; set; }
    public byte G { get; set; }
    public byte B { get; set; }
    public bool Active { get; set; }

    public Ball(double x, double y, double vx, double vy, double radius, byte r, byte g, byte b)
    {
        X = x;
        Y = y;
        Vx = vx;
        Vy = vy;
        Radius = radius;
        R = r;
        G = g;
        B = b;
        Active = true;
    }
}
