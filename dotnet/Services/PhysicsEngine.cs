using DotnetApi.Models;

namespace DotnetApi.Services;

public class PhysicsEngine
{
    private readonly SimulationConfig _config;
    private readonly List<SurfacePoint> _surface;
    private readonly int _canvasWidth;
    private readonly int _canvasHeight;

    public PhysicsEngine(SimulationConfig config, List<SurfacePoint> surface, int canvasWidth, int canvasHeight)
    {
        _config = config;
        _surface = surface;
        _canvasWidth = canvasWidth;
        _canvasHeight = canvasHeight;
    }

    public SimulationState Update(Ball[] balls)
    {
        var gravity = _config.Gravity * 100; // Scale gravity for pixel coordinates
        var deltaTime = _config.DeltaTime;
        var restitution = _config.Restitution;
        var airResistance = _config.AirResistance;

        Parallel.For(0, balls.Length, i =>
        {
            if (!balls[i].Active) return;

            var ball = balls[i];

            // Apply gravity
            ball.Vy += gravity * deltaTime;

            // Apply air resistance
            ball.Vx *= (1 - airResistance);
            ball.Vy *= (1 - airResistance);

            // Update position
            ball.X += ball.Vx * deltaTime;
            ball.Y += ball.Vy * deltaTime;

            // Check surface collision
            CheckSurfaceCollision(ref ball, restitution);

            // Check bounds (off-screen)
            if (ball.X < -50 || ball.X > _canvasWidth + 50 || 
                ball.Y < -50 || ball.Y > _canvasHeight + 50)
            {
                ball.Active = false;
            }

            balls[i] = ball;
        });

        return new SimulationState
        {
            Balls = balls,
            Timestamp = DateTime.UtcNow.Ticks
        };
    }

    private void CheckSurfaceCollision(ref Ball ball, double restitution)
    {
        if (_surface.Count < 2) return;

        for (int i = 0; i < _surface.Count - 1; i++)
        {
            var p1 = _surface[i];
            var p2 = _surface[i + 1];

            // Calculate distance from ball to line segment
            var dx = p2.X - p1.X;
            var dy = p2.Y - p1.Y;
            var lengthSq = dx * dx + dy * dy;

            if (lengthSq == 0) continue;

            // Project ball onto line segment
            var t = Math.Max(0, Math.Min(1, ((ball.X - p1.X) * dx + (ball.Y - p1.Y) * dy) / lengthSq));

            var closestX = p1.X + t * dx;
            var closestY = p1.Y + t * dy;

            var distX = ball.X - closestX;
            var distY = ball.Y - closestY;
            var distance = Math.Sqrt(distX * distX + distY * distY);

            if (distance < ball.Radius)
            {
                // Calculate collision normal
                var normalX = distX / distance;
                var normalY = distY / distance;

                // Reflect velocity: V = V - 2(V·N)N
                var dotProduct = ball.Vx * normalX + ball.Vy * normalY;
                ball.Vx = (ball.Vx - 2 * dotProduct * normalX) * restitution;
                ball.Vy = (ball.Vy - 2 * dotProduct * normalY) * restitution;

                // Push ball out of surface
                ball.X = closestX + normalX * (ball.Radius + 0.1);
                ball.Y = closestY + normalY * (ball.Radius + 0.1);
            }
        }
    }
}
