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
        builder.AllowAnyOrigin()
               .AllowAnyMethod()
               .AllowAnyHeader();
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
