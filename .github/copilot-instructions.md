# Project: MyWebsite Portfolio

## Architecture

**Dev (single container):** All services run inside the devcontainer:
```
frontend/        → React SPA (Vite, port 3000) — runs via VS Code task
backend/         → Flask/Python ML API (port 5000) — runs via VS Code task
dotnet/          → ASP.NET Core 8.0 Web API (port 5001) — runs via VS Code task
```
All three services start automatically via VS Code tasks when the workspace folder opens (configured in `.vscode/tasks.json`).

**Prod (separate containers):** Root `Dockerfile` + `docker-compose.yml` builds separate images for each service.

## Key Paths

- Frontend source: `frontend/src/`
- Backend API: `backend/api/api.py` + `backend/api/routes/`
- ML code: `backend/api/src/machine_learning/`
- .NET API: `dotnet/Controllers/`
- Dev container: `.devcontainer/` (single Dockerfile, no docker-compose)
- Prod Docker: root `Dockerfile` + `docker-compose.yml`
- Logs: `logs/`

## Running

- **Dev (recommended):** Reopen in devcontainer. All 3 services start automatically via VS Code tasks (Flask 5000, dotnet 5001, React 3000).
- **Prod:** `docker-compose up site dotnet-api` from root.
- **Restart services:** Use `workbench.action.tasks.restartTask` with args `["Task Name"]` to restart a task cleanly (no UI prompt)

## Logs

- Logs of the running tasks are populated from the output of the running tasks in the `logs/` folder, filenames like `task-*.log`
- Server logs of the dotnet service are stored in `logs/dotnet-*.log` files, with todays date.
- IMPORTANT: Always check the logs when debugging work related to the services. Don't assume things are working.

## Frontend

- React 18 with Vite (port 3000, host: '0.0.0.0').
- React Router v5 (`react-router-dom` BrowserRouter).
- Pages: `/`, `/home`, `/machinelearner`, `/pathfinder`, `/dotnet-demo`.
- Components live in `frontend/src/components/`.
- CSS is per-component (module-style `.css` files).
- Runs using task "Start React Dev Server (Port 3000)".

## Backend (Flask)

- `FLASK_APP=backend/api/api.py` (devcontainer).
- All ML routes are under `/api/ml*` via blueprint registration.
- Core ML entry: `backend/api/src/machine_learning/MachineLearner_Functions.py`.
- ML model types: `backend/api/src/machine_learning/models/model_types/` (decision_tree, k_nearest_neighbours, linear_regression, polynomial_regression).
- CORS enabled globally via `flask-cors`.
- Session-based ML state managed via `session_context_manager.py` — always validate session IDs exist before accessing.
- Runs using task "Start Flask Server (Port 5000)".

## Backend (.NET)

- ASP.NET Core 8.0 Web API.
- Route pattern: `/api/[controller]/[action]` (e.g., `/api/example/hello`).
- CORS configured with "AllowAll" policy in `Program.cs`.
- Controllers go in `dotnet/Controllers/`.
- Runs using task "Start dotnet Service (Port 5001)".

## Docker / Devcontainer

- Devcontainer uses `docker-compose.yml` (single `site-dev` service with all components).
- Volume mounts for hot reload: `./frontend`, `./backend`, `./dotnet`, `./notes` → `/app/`
- Build artifact volumes (persist across rebuilds):
  - `frontend-node_modules` → `/app/frontend/node_modules`
  - `frontend-build` → `/app/frontend/build`
  - `dotnet-obj` → `/app/dotnet/obj`
  - `dotnet-bin` → `/app/dotnet/bin`
  - `backend-pycache` → `/app/backend/__pycache__`
- All services on `mywebsite-network` bridge network.
- `.dockerignore` excludes build artifacts from Docker COPY commands.

## Conventions

- No TypeScript — all JS.
- No CSS-in-JS — separate `.css` files with component-level scope.
- Flask blueprints for route organization (`backend/api/routes/`).
- .NET controllers follow standard ASP.NET Core conventions.
- Always check if session data exists before accessing it in ML routes (raises `KeyError` otherwise).
- Keep devcontainer compose file separate from prod (`docker-compose.yml`) — never reference the root file from `devcontainer.json`.
- Notes and non-code files go in `notes/` directory.
