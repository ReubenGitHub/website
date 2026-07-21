namespace DotnetApi.Services;
using DotnetApi.Models;

/// <summary>
/// Spatial hash grid for accelerating surface segment lookups during collision detection.
/// Pre-computes which surface segments pass through each grid cell, so collision detection
/// only checks segments near the ball instead of iterating the entire surface.
/// 
/// Reduces complexity from O(balls × surface_points × substeps) to
/// O(balls × avg_collisions_per_cell × substeps) — typically 10-50x faster.
/// </summary>
public sealed class SpatialHashGrid
{
    private readonly int _cellSize;
    private readonly Dictionary<(int CellX, int CellY), List<int>> _grid;

    /// <summary>
    /// Creates a spatial hash grid for the given surface.
    /// </summary>
    /// <param name="surface">Ordered list of surface points defining line segments.</param>
    /// <param name="cellSize">Grid cell size in pixels. Must be positive.</param>
    public SpatialHashGrid(List<SurfacePoint> surface, int cellSize)
    {
        _cellSize = cellSize;
        _grid = new Dictionary<(int, int), List<int>>();

        for (int i = 0; i < surface.Count - 1; i++)
        {
            var p1 = surface[i];
            var p2 = surface[i + 1];

            // Determine which cells this segment passes through
            var minX = Math.Min(p1.X, p2.X);
            var maxX = Math.Max(p1.X, p2.X);
            var minY = Math.Min(p1.Y, p2.Y);
            var maxY = Math.Max(p1.Y, p2.Y);

            var cellMinX = (int)Math.Floor(minX / _cellSize);
            var cellMaxX = (int)Math.Floor(maxX / _cellSize);
            var cellMinY = (int)Math.Floor(minY / _cellSize);
            var cellMaxY = (int)Math.Floor(maxY / _cellSize);

            for (int cy = cellMinY; cy <= cellMaxY; cy++)
            {
                for (int cx = cellMinX; cx <= cellMaxX; cx++)
                {
                    if (!_grid.TryGetValue((cx, cy), out var segments))
                    {
                        segments = new List<int>(4);
                        _grid[(cx, cy)] = segments;
                    }
                    segments.Add(i);
                }
            }
        }
    }

        /// <summary>
        /// Gets the segment indices that are in the cells overlapping the given rectangle.
        /// </summary>
        /// <param name="minX">Left edge of the query rectangle.</param>
        /// <param name="minY">Top edge of the query rectangle.</param>
        /// <param name="maxX">Right edge of the query rectangle.</param>
        /// <param name="maxY">Bottom edge of the query rectangle.</param>
        /// <param name="buffer">Extra buffer to add around the rectangle (e.g., ball radius).</param>
        /// <returns>Enumerable of segment indices to check (no intermediate allocation).</returns>
        public IEnumerable<int> GetSegmentsInRect(double minX, double minY, double maxX, double maxY, double buffer = 0)
        {
            // Expand query rectangle by buffer
            minX -= buffer;
            minY -= buffer;
            maxX += buffer;
            maxY += buffer;

            var cellMinX = (int)Math.Floor(minX / _cellSize);
            var cellMaxX = (int)Math.Floor(maxX / _cellSize);
            var cellMinY = (int)Math.Floor(minY / _cellSize);
            var cellMaxY = (int)Math.Floor(maxY / _cellSize);

            for (int cy = cellMinY; cy <= cellMaxY; cy++)
            {
                for (int cx = cellMinX; cx <= cellMaxX; cx++)
                {
                    if (_grid.TryGetValue((cx, cy), out var segments))
                    {
                        foreach (var seg in segments)
                            yield return seg;
                    }
                }
            }
        }
}
