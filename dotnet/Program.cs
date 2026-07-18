using DotnetApi.Hubs;
using DotnetApi.Services;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container
builder.Services.AddControllers();
builder.Services.AddSignalR();
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowAll", builder =>
    {
        // Reflect the requesting origin instead of wildcard '*'
        // Required for SignalR which sends credentials by default
        builder.SetIsOriginAllowed(origin =>
                origin != null && (origin.Contains("localhost") || origin.Contains("127.0.0.1")))
               .AllowAnyMethod()
               .AllowAnyHeader()
               .AllowCredentials();
    });
});

// Register simulation services (scoped per connection)
builder.Services.AddScoped<SurfaceService>();
builder.Services.AddScoped<SimulationSession>();

var app = builder.Build();

// Configure the HTTP request pipeline
if (app.Environment.IsDevelopment())
{
    app.UseDeveloperExceptionPage();
}

app.UseHttpsRedirection();
app.UseCors("AllowAll");
app.UseAuthorization();
app.MapControllers();

// Map SignalR hub
app.MapHub<PhysicsHub>("/physicsHub");

// Health check endpoint
app.MapGet("/health", () =>
{
    return Results.Ok(new { status = "healthy", service = "dotnet-api" });
});

app.Run();
