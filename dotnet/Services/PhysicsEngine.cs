using DotnetApi.Models;

namespace DotnetApi.Services;

public class PhysicsEngine
{
    private readonly SimulationConfig _config;
    private readonly List<SurfacePoint> _surface;
    private readonly int _canvasWidth;
    private readonly int _canvasHeight;
    private readonly SpatialHashGrid _hashGrid;
    
    // Spatial hash grid cell size (matches ball diameter for efficient lookups)
    private const int CellSize = 5;
    
    // Sub-stepping: split each frame into smaller steps to prevent tunneling
    // 16 sub-steps with velocity clamping prevents tunneling while maintaining performance
    private const int SubSteps = 16;
    
    // Max velocity clamp to prevent tunneling (px/s) — ~2x max fall speed from canvas top
    private const double MaxVelocity = 1000.0;

    public PhysicsEngine(SimulationConfig config, List<SurfacePoint> surface, int canvasWidth, int canvasHeight)
    {
        _config = config;
        _surface = surface;
        _canvasWidth = canvasWidth;
        _canvasHeight = canvasHeight;
        
        // Pre-compute spatial hash grid for fast surface segment lookups
        _hashGrid = new SpatialHashGrid(surface, CellSize);
    }

    public SimulationState Update(Ball[] balls)
    {
        var gravity = _config.Gravity * 100; // Scale gravity for pixel coordinates
        var deltaTime = _config.DeltaTime;
        var restitution = _config.Restitution;
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

                // Clamp velocity to prevent tunneling
                var speed = Math.Sqrt(ball.Vx * ball.Vx + ball.Vy * ball.Vy);
                if (speed > MaxVelocity)
                {
                    var scale = MaxVelocity / speed;
                    ball.Vx *= scale;
                    ball.Vy *= scale;
                }

                // Update position
                ball.X += ball.Vx * subDeltaTime;
                ball.Y += ball.Vy * subDeltaTime;

                // Simple overlap collision detection
                CheckSurfaceCollision(ref ball, restitution, 0, subDeltaTime);
            }

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

    private bool CheckSurfaceCollision(ref Ball ball, double restitution, double gravity, double subDeltaTime)
    {
        if (_surface.Count < 2) return false;

        bool collisionOccurred = false;

        // Use spatial hash grid to only check segments near the ball
        var ballMinX = ball.X - ball.Radius;
        var ballMinY = ball.Y - ball.Radius;
        var ballMaxX = ball.X + ball.Radius;
        var ballMaxY = ball.Y + ball.Radius;

        foreach (var segIndex in _hashGrid.GetSegmentsInRect(ballMinX, ballMinY, ballMaxX, ballMaxY, 0))
        {
            var p1 = _surface[segIndex];
            var p2 = _surface[segIndex + 1];

            // Calculate distance from ball to line segment
            var dx = p2.X - p1.X;
            var dy = p2.Y - p1.Y;
            var lengthSq = dx * dx + dy * dy;

            if (lengthSq == 0) continue;

            // Quick bounding box check — skip if ball is clearly too far from segment
            var minX = Math.Min(p1.X, p2.X) - ball.Radius;
            var maxX = Math.Max(p1.X, p2.X) + ball.Radius;
            var minY = Math.Min(p1.Y, p2.Y) - ball.Radius;
            var maxY = Math.Max(p1.Y, p2.Y) + ball.Radius;
            if (ball.X < minX || ball.X > maxX || ball.Y < minY || ball.Y > maxY)
                continue;

            // Project ball onto line segment
            var t = Math.Max(0, Math.Min(1, ((ball.X - p1.X) * dx + (ball.Y - p1.Y) * dy) / lengthSq));

            var closestX = p1.X + t * dx;
            var closestY = p1.Y + t * dy;

            var distX = ball.X - closestX;
            var distY = ball.Y - closestY;
            var distSq = distX * distX + distY * distY;
            var radiusSq = ball.Radius * ball.Radius;

            // Squared distance check — avoids expensive Math.Sqrt for far-away segments
            if (distSq >= radiusSq) continue;

            var distance = Math.Sqrt(distSq);

            // Calculate collision normal (from surface to ball)
            var normalX = distX / Math.Max(0.001, distance);
            var normalY = distY / Math.Max(0.001, distance);

            // Snap normal to vertical for near-horizontal surfaces.
            // This prevents systematic sideways drift caused by floating-point
            // imprecision in the normal calculation when balls land near segment
            // endpoints on flat surfaces. A segment is "near-horizontal" when its
            // slope is less than ~5 degrees (|dy|/|dx| < 0.087).
            var segDy = dy / Math.Max(Math.Abs(dx), Math.Abs(dy));
            if (Math.Abs(segDy) < 0.087)
            {
                // Force normal to point straight up (normalY < 0 in canvas coords)
                normalX = 0;
                normalY = Math.Sign(normalY);
            }

            // In canvas coords, Y increases downward.
            // If normalY < 0, the normal points upward, meaning ball is above surface.
            // If normalY > 0, the normal points downward, meaning ball is below surface.
            // Only process collision if ball is above the surface.
            if (normalY > -0.1) continue;

            // Check if ball is moving toward the surface (velocity opposite to normal)
            var dotProduct = ball.Vx * normalX + ball.Vy * normalY;
            
            // Only reflect if ball is moving toward the surface (dotProduct < 0 means velocity opposes normal)
            // Reflect velocity: V = V - 2(V·N)N
            if (dotProduct < 0)
            {
                // Reflect velocity: V = V - 2(V·N)N
                ball.Vx = (ball.Vx - 2 * dotProduct * normalX) * _config.Restitution;
                ball.Vy = (ball.Vy - 2 * dotProduct * normalY) * _config.Restitution;
            }

            // Push ball out of surface to prevent sinking (use exact radius, no extra offset)
            ball.X = closestX + normalX * ball.Radius;
            ball.Y = closestY + normalY * ball.Radius;
                
            collisionOccurred = true;
        }

        return collisionOccurred;
    }

}
