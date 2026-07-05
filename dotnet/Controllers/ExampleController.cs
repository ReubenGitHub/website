using Microsoft.AspNetCore.Mvc;

namespace DotNetApi.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class ExampleController : ControllerBase
    {
        [HttpGet("hello")]
        public ActionResult<object> GetHello()
        {
            return Ok(new
            {
                message = "Hello from C# .NET API!",
                timestamp = DateTime.UtcNow,
                service = "dotnet-api"
            });
        }

        [HttpPost("echo")]
        public ActionResult<object> PostEcho([FromBody] EchoRequest? request)
        {
            if (request == null)
            {
                return BadRequest(new { error = "Request body is required" });
            }

            return Ok(new
            {
                echoed_message = request.Message,
                processed_at = DateTime.UtcNow,
                length = request.Message?.Length ?? 0
            });
        }

        [HttpGet("data")]
        public ActionResult<object> GetData()
        {
            return Ok(new
            {
                items = new[]
                {
                    new { id = 1, name = "Item 1", value = 100 },
                    new { id = 2, name = "Item 2", value = 200 },
                    new { id = 3, name = "Item 3", value = 300 }
                },
                count = 3,
                timestamp = DateTime.UtcNow
            });
        }
    }

    public class EchoRequest
    {
        public string? Message { get; set; }
    }
}
