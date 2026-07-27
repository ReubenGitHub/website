using DotnetApi.Hubs;
using DotnetApi.Services;
using Serilog;
using AWS.Logger;
using AWS.Logger.SeriLog;

var builder = WebApplication.CreateBuilder(args);

// Add Serilog logging with environment-aware configuration
var minimumLevel = builder.Environment.IsDevelopment()
    ? Serilog.Events.LogEventLevel.Debug
    : Serilog.Events.LogEventLevel.Information;

var loggerConfig = new LoggerConfiguration()
    .MinimumLevel.Is(minimumLevel);

// In production, write to CloudWatch Logs and file
// In development, write to file only
if (builder.Environment.IsProduction())
{
    var awsConfig = new AWSLoggerConfig("/mywebsite/dotnet-api")
    {
        Region = "eu-west-2"
    };
    
    loggerConfig
        .WriteTo.AWSSeriLog(awsConfig);
}

loggerConfig
    .WriteTo.File(
        path: "/app/logs/dotnet-.log",
        rollingInterval: RollingInterval.Day,
        outputTemplate: "{Timestamp:yyyy-MM-dd HH:mm:ss.fff zzz} [{Level:u3}] {Message:lj}{NewLine}{Exception}",
        retainedFileCountLimit: 31);

Log.Logger = loggerConfig.CreateLogger();

builder.Host.UseSerilog();

// Also add console logging via ASP.NET Core logging (for awslogs driver)
builder.Logging.AddConsole();

// Add services to the container
builder.Services.AddControllers()
    .AddJsonOptions(options =>
    {
        options.JsonSerializerOptions.PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase;
    });
builder.Services.AddSignalR(options =>
{
    // Increase max message size to accommodate spawn mask data (1200x600 = 720KB)
    options.MaximumReceiveMessageSize = 4 * 1024 * 1024; // 4MB
});

// Configure CORS based on environment
// Production: Read from environment variables (DOMAIN, CORS_ORIGINS)
// Development: Allow localhost
string[] allowedOrigins;
if (builder.Environment.IsProduction())
{
    var domain = builder.Configuration.GetValue<string>("DOMAIN") ?? "reubenhow.com";
    var corsOrigins = builder.Configuration.GetValue<string>("CORS_ORIGINS");
    
    if (!string.IsNullOrEmpty(corsOrigins))
    {
        // Allow multiple origins separated by commas
        allowedOrigins = corsOrigins.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
    }
    else
    {
        // Default to domain-based origins
        allowedOrigins = [$"https://{domain}", $"https://www.{domain}"];
    }
}
else
{
    allowedOrigins = ["http://localhost:3000", "http://[::1]:3000"];
}

builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowFrontend", policy =>
    {
        policy.WithOrigins(allowedOrigins)
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

app.UseCors("AllowFrontend");
app.UseAuthorization();
app.MapControllers();

// Map SignalR hub
app.MapHub<PhysicsHub>("/physicsHub");

// Health check endpoint
app.MapGet("/health", () => Results.Ok(new { status = "healthy", service = "dotnet-api" }));

app.Run();
