# Project: MyWebsite Portfolio

## Architecture

Three-service microservices architecture:

```
frontend/        → React SPA (CRA)
backend/         → Flask/Python ML API (port 5000)
dotnet/          → ASP.NET Core microservice (port 5001)
```

The Flask backend serves the React build as static files. The .NET service is an independent REST API. All services communicate via Docker network `mywebsite-network`.

## Key Paths

- Frontend source: `frontend/src/`
- Backend API: `backend/api/api.py` + `backend/api/routes/`
- ML code: `backend/api/src/machine_learning/`
- .NET API: `dotnet/Controllers/`
- Dev container: `.devcontainer/`
- Prod Docker: root `Dockerfile` + `docker-compose.yml`

## Running

- **Dev (recommended):** Reopen in devcontainer. Runs `docker-compose -f .devcontainer/docker-compose.yml up`.
- **Prod:** `docker-compose up site dotnet-api` from root.

## Frontend

- React 18 with Create React App (ejected-free — uses `react-scripts`).
- React Router v5 (`react-router-dom` BrowserRouter).
- Pages: `/`, `/home`, `/machinelearner`, `/pathfinder`, `/dotnet-demo`.
- Components live in `frontend/src/components/`.
- CSS is per-component (module-style `.css` files).
- Frontend is built (`npm run build`) and served by Flask as static files.

## Backend (Flask)

- `FLASK_APP=backend/api/api.py` (devcontainer).
- All ML routes are under `/api/ml*` via blueprint registration.
- Core ML entry: `backend/api/src/machine_learning/MachineLearner_Functions.py`.
- ML model types: `backend/api/src/machine_learning/models/model_types/` (decision_tree, k_nearest_neighbours, linear_regression, polynomial_regression).
- CORS enabled globally via `flask-cors`.
- Session-based ML state managed via `session_context_manager.py` — always validate session IDs exist before accessing.

## Backend (.NET)

- ASP.NET Core 8.0 Web API.
- Route pattern: `/api/[controller]/[action]` (e.g., `/api/example/hello`).
- CORS configured with "AllowAll" policy in `Program.cs`.
- Controllers go in `dotnet/Controllers/`.
- Use `dotnet watch run` for dev hot reload.

## Docker / Devcontainer

- Devcontainer uses `.devcontainer/docker-compose.yml` (2 services only: `site-dev`, `dotnet-api-dev`).
- Backend volume mount: `../backend:/app/backend` (hot reload for Python code).
- Frontend build is baked into the image (postStartCommand runs `npm run build`).
- .NET volume mount: `../dotnet:/src` (hot reload for C# code).
- Both services on `mywebsite-network` bridge network.

## Conventions

- No TypeScript — all JS.
- No CSS-in-JS — separate `.css` files with component-level scope.
- Flask blueprints for route organization (`backend/api/routes/`).
- .NET controllers follow standard ASP.NET Core conventions.
- Always check if session data exists before accessing it in ML routes (raises `KeyError` otherwise).
- Keep devcontainer compose file separate from prod (`docker-compose.yml`) — never reference the root file from `devcontainer.json`.
- Notes and non-code files go in `notes/` directory.
