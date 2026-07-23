using DotnetApi.Hubs;
using DotnetApi.Services;
using Serilog;

var builder = WebApplication.CreateBuilder(args);

// Add Serilog file logging
Log.Logger = new LoggerConfiguration()
    .MinimumLevel.Debug()
    .WriteTo.File(
        path: "/app/logs/dotnet-.log",
        rollingInterval: RollingInterval.Day,
        outputTemplate: "{Timestamp:yyyy-MM-dd HH:mm:ss.fff zzz} [{Level:u3}] {Message:lj}{NewLine}{Exception}",
        retainedFileCountLimit: 31)
    .CreateLogger();

builder.Host.UseSerilog();

// Add services to the container
builder.Services.AddControllers()
    .AddJsonOptions(options =>
    {
        options.JsonSerializerOptions.PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase;
    });
builder.Services.AddSignalR();
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowAll", policy =>
    {
        policy.WithOrigins("http://localhost:3000", "http://[::1]:3000")
              .AllowAnyMethod()
              .AllowAnyHeader()
              .AllowCredentials();
    });
});

// Register simulation services
builder.Services.AddSingleton<SurfaceService>();
builder.Services.AddSingleton<SimulationSession>();
builder.Services.AddSingleton<SimulationStreamService>();
builder.Services.AddHostedService(provider => provider.GetRequiredService<SimulationStreamService>());

var app = builder.Build();

// Configure the HTTP request pipeline
if (app.Environment.IsDevelopment())
{
    app.UseDeveloperExceptionPage();
}

app.UseCors("AllowAll");
app.UseAuthorization();
app.MapControllers();

// Map SignalR hub
app.MapHub<PhysicsHub>("/physicsHub");

// Health check endpoint
app.MapGet("/health", () => Results.Ok(new { status = "healthy", service = "dotnet-api" }));

app.Run();
