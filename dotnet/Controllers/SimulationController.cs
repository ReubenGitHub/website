using Microsoft.AspNetCore.Mvc;
using DotnetApi.Models;
using DotnetApi.Services;

namespace DotnetApi.Controllers;

[ApiController]
[Route("api/[controller]")]
public class SimulationController : ControllerBase
{
    private readonly ILogger<SimulationController> _logger;

    public SimulationController(ILogger<SimulationController> logger)
    {
        _logger = logger;
    }

    [HttpGet("health")]
    public IActionResult GetHealth()
    {
        return Ok(new { status = "healthy", service = "physics-simulation" });
    }

    [HttpGet("config")]
    public IActionResult GetConfig()
    {
        var config = new SimulationConfig
        {
            BallCount = 2000,
            Gravity = 9.8,
            Restitution = 0.7,
            AirResistance = 0.01,
            DeltaTime = 1.0 / 30.0
        };
        return Ok(config);
    }

    [HttpPost("validate")]
    public IActionResult ValidateConfig([FromBody] SimulationConfig config)
    {
        if (config == null)
            return BadRequest(new { error = "Config is required" });

        if (config.BallCount < 10 || config.BallCount > 10000)
            return BadRequest(new { error = "Ball count must be between 10 and 10000" });

        if (config.Gravity <= 0)
            return BadRequest(new { error = "Gravity must be positive" });

        if (config.Restitution < 0 || config.Restitution > 1)
            return BadRequest(new { error = "Restitution must be between 0 and 1" });

        return Ok(new { valid = true });
    }
}
