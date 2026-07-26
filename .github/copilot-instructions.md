# Project: MyWebsite Portfolio

## Architecture

**Dev (single container):** All services run inside the devcontainer:
```
frontend/        → React SPA (Vite, port 3000) — runs via VS Code task
backend/         → Flask/Python ML API (port 5000) — runs via VS Code task
dotnet/          → ASP.NET Core 8.0 Web API (port 5001) — runs via VS Code task
```
All three services start automatically via VS Code tasks when the workspace folder opens (configured in `.vscode/tasks.json`).

**Prod (AWS ECS Fargate):** Single Fargate task with 3 containers, deployed via `deployment/deploy.sh`

## Key Paths

- Frontend source: `frontend/src/`
- Backend API: `backend/api/api.py` + `backend/api/routes/`
- ML code: `backend/api/src/machine_learning/`
- .NET API: `dotnet/Controllers/`
- Dev container: `.devcontainer/` (single Dockerfile, no docker-compose)
- Prod deployment: `deployment/` (Dockerfiles, deploy.sh, ecs-task-definition.json)
- Logs: `logs/` (dev), CloudWatch `/mywebsite/{frontend,backend,dotnet-api}` (prod)

## Running

- **Dev (recommended):** Reopen in devcontainer. All 3 services start automatically via VS Code tasks (Flask 5000, dotnet 5001, React 3000).
- **Prod:** `bash deployment/deploy.sh` (full deploy) or individual steps:
  - `./deploy.sh setup` — Create AWS resources (ECR, ECS, IAM, ALB, ACM, log groups)
  - `./deploy.sh build` — Build and push Docker images to ECR
  - `./deploy.sh deploy` — Register task definition + update ECS service
- **Restart prod:** `aws ecs update-service --cluster mywebsite-production --service mywebsite-production --task-definition mywebsite-production:<VERSION> --force-new-deployment --desired-count 1 --region eu-west-2`
- **Restart dev services:** Use `workbench.action.tasks.restartTask` with args `["Task Name"]` to restart a task cleanly (no UI prompt)

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
- Route pattern: `/api/[controller]/[action]` (e.g., `/api/simulation/health`).
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

## Devcontainer Tool Persistence
- CRITICAL: When installing system packages (apt-get, npm, pip, etc.) inside the devcontainer, ALWAYS ensure they are added to `.devcontainer/Dockerfile` for persistence across rebuilds.
- Never rely on `apt-get install` or similar commands run in the terminal alone — those installs are lost on devcontainer rebuild.

## Production Deployment (AWS ECS Fargate)

**Architecture:** Single Fargate task, 3 containers, awsvpc network mode
- Frontend: Nginx (port 80) — serves React SPA + proxies /api/* and /physicsHub
- Backend: Flask (port 5000) — ML API
- Dotnet-API: ASP.NET Core (port 5001) — physics simulation + REST API
- All containers communicate via localhost (same task)
- ALB routes to frontend container, ECS auto-manages target group

**AWS Details:**
- Region: eu-west-2, Profile: mywebsite-production (account ID in aws.config, gitignored)
- Cluster: mywebsite-production, Service: mywebsite-production
- ALB: mywebsite-alb, Target Group: mywebsite-frontend-tg
- VPC: vpc-092c08b3956a1080d (default), Subnets: eu-west-2a/b/c
- Security Group: mywebsite-alb-sg
- ECR: <account-id>.dkr.ecr.eu-west-2.amazonaws.com/reubenhow (resolved dynamically by deploy.sh)
- Domain: reubenhow.com (Cloudflare, Flexible SSL)

**Deployment Files:**
- `deployment/deploy.sh` — Main deployment script (setup/build/deploy)
- `deployment/Dockerfile.frontend` — Frontend build (uses frontend/nginx.conf)
- `deployment/Dockerfile.flask` — Backend build
- `deployment/Dockerfile.dotnet` — Dotnet build
- `deployment/ecs-task-definition.json` — Task def template (uses :latest tags)
- `deployment/.env` — AWS credentials (from AWS-SSO-SETUP.md)
- `frontend/nginx.conf` — Nginx config (WebSocket support for /physicsHub)

**How deploy.sh works:**
1. Task definition uses `:latest` tags for all containers
2. deploy.sh replaces `:latest` with git hash when registering task definition
3. Images pushed as both `:<hash>` and `:latest` in ECR
4. ECS uses the git hash-tagged images (pinned to specific build)

**Nginx WebSocket Config:**
- `/physicsHub` → localhost:5001 (SignalR hub)
- `/api/simulation/*` → localhost:5001 (.NET API)
- `/api/*` → localhost:5000 (Flask)

**Troubleshooting:**
- Check CloudWatch logs: `/mywebsite/frontend`, `/mywebsite/backend`, `/mywebsite/dotnet-api`
- Use `aws logs get-log-events` with specific log stream, NEVER `aws logs tail --follow` (blocks)
- Task crashes: check exit codes (137=OOM, 143=SIGTERM, 1=generic error)
- Health checks: frontend=/, backend=/api/time, dotnet=/api/simulation/health
- Always use `--output json` with AWS CLI, never `--output table` (pager issues)
- Set `AWS_PAGER=""` in scripts to avoid pager

**Current Task Config:**
- CPU: 512, Memory: 1024 MB (user reverted from 1024/2048)
- Images: frontend:latest, backend:latest, dotnet-api:latest (in task def)
- Latest stable task version: check with `aws ecs list-task-definitions`
