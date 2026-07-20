using DotnetApi.Models;

namespace DotnetApi.Services;

public class PhysicsEngine
{
    private readonly SimulationConfig _config;
    private readonly List<SurfacePoint> _surface;
    private readonly int _canvasWidth;
    private readonly int _canvasHeight;
    
    // Sub-stepping: split each frame into smaller steps to prevent tunneling
    private const int SubSteps = 8;
    // Velocity threshold below which a ball is considered resting on surface
    private const double RestingVelocityThreshold = 8.0;
    // Gravity threshold: if ball is near surface and moving slowly, settle it
    private const double SettleGravityThreshold = 1.0;
    // Maximum distance per sub-step (prevents tunneling - must be < ball radius)
    private const double MaxDistancePerStep = 1.5;

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
        var subDeltaTime = deltaTime / SubSteps;

        Parallel.For(0, balls.Length, i =>
        {
            if (!balls[i].Active) return;

            var ball = balls[i];

            // Sub-stepping: run multiple smaller physics steps per frame
            for (int step = 0; step < SubSteps; step++)
            {
                // Apply gravity
                ball.Vy += gravity * subDeltaTime;

                // Apply air resistance
                var airFactor = 1 - (airResistance / SubSteps);
                ball.Vx *= airFactor;
                ball.Vy *= airFactor;

                // Clamp velocity to prevent tunneling through surface
                // Max distance per sub-step must be less than ball radius
                var maxSpeed = MaxDistancePerStep / subDeltaTime;
                var speed = Math.Sqrt(ball.Vx * ball.Vx + ball.Vy * ball.Vy);
                if (speed > maxSpeed)
                {
                    var scale = maxSpeed / speed;
                    ball.Vx *= scale;
                    ball.Vy *= scale;
                }

                // Update position
                ball.X += ball.Vx * subDeltaTime;
                ball.Y += ball.Vy * subDeltaTime;

                // Check surface collision
                CheckSurfaceCollision(ref ball, restitution, gravity, subDeltaTime);
            }

            // Check if ball is resting on surface (low velocity + near surface)
            CheckRestingState(ref ball, restitution);

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

    private void CheckRestingState(ref Ball ball, double restitution)
    {
        if (_surface.Count < 2) return;

        var speed = Math.Sqrt(ball.Vx * ball.Vx + ball.Vy * ball.Vy);
        
        // If ball is moving very slowly, check if it's resting on surface
        if (speed < RestingVelocityThreshold)
        {
            for (int i = 0; i < _surface.Count - 1; i++)
            {
                var p1 = _surface[i];
                var p2 = _surface[i + 1];

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

                // If ball is very close to surface
                if (distance < ball.Radius + MaxDistancePerStep)
                {
                    // Calculate surface normal (from surface to ball)
                    var normalX = distX / Math.Max(0.001, distance);
                    var normalY = distY / Math.Max(0.001, distance);

                    // In canvas coords, Y increases downward.
                    // If normalY < 0, the normal points upward, meaning ball is above surface.
                    // Only settle if ball is above the surface
                    if (normalY < -0.1 && ball.Vy > -RestingVelocityThreshold)
                    {
                        // Set velocity to zero
                        ball.Vx = 0;
                        ball.Vy = 0;
                        
                        // Position ball exactly on surface
                        ball.X = closestX + normalX * (ball.Radius + 0.5);
                        ball.Y = closestY + normalY * (ball.Radius + 0.5);
                        return;
                    }
                }
            }
        }
    }

    private void CheckSurfaceCollision(ref Ball ball, double restitution, double gravity, double subDeltaTime)
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

            if (distance < ball.Radius + MaxDistancePerStep)
            {
                // Calculate collision normal (from surface to ball)
                var normalX = distX / Math.Max(0.001, distance);
                var normalY = distY / Math.Max(0.001, distance);

                // In canvas coords, Y increases downward.
                // If normalY < 0, the normal points upward, meaning ball is above surface.
                // If normalY > 0, the normal points downward, meaning ball is below surface.
                // Only process collision if ball is above the surface.
                if (normalY > -0.1) continue;

                // Check if ball is moving toward the surface (velocity opposite to normal)
                var dotProduct = ball.Vx * normalX + ball.Vy * normalY;
                
                // Only reflect if ball is moving toward the surface (dotProduct < 0 means velocity opposes normal)
                if (dotProduct < 0)
                {
                    // Reflect velocity: V = V - 2(V·N)N
                    ball.Vx = (ball.Vx - 2 * dotProduct * normalX) * restitution;
                    ball.Vy = (ball.Vy - 2 * dotProduct * normalY) * restitution;
                }

                // Push ball out of surface to prevent sinking
                ball.X = closestX + normalX * (ball.Radius + 0.5);
                ball.Y = closestY + normalY * (ball.Radius + 0.5);
            }
        }
    }
}
