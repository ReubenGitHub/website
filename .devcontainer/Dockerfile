ARG NODE_VERSION_NUMBER=22
ARG PYTHON_VERSION_NUMBER=3.9
ARG PYTHON_IMAGE_VERSION=${PYTHON_VERSION_NUMBER}-slim

FROM python:${PYTHON_IMAGE_VERSION}
ARG NODE_VERSION_NUMBER

WORKDIR /app

# Create directories for all project components
RUN mkdir -p /app/backend /app/frontend /app/dotnet /app/notes /app/logs

# Copy package files for dependency installation
COPY frontend/package.json /app/frontend/
COPY backend/requirements.txt /app/backend/

# Install build tools, Node.js, and .NET SDK
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        git \
        graphviz \
        curl \
        gnupg \
        ca-certificates \
        apt-transport-https \
        jq && \
    curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION_NUMBER}.x | bash - && \
    apt-get install -y nodejs && \
    # Install .NET SDK 8.0 using official install script (avoids apt repo signing issues on Debian 13)
    curl -fsSL https://dot.net/v1/dotnet-install.sh | bash /dev/stdin --channel 8.0 --version 8.0.404 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY frontend/ /app/frontend/
COPY backend/ /app/backend/
COPY dotnet/ /app/dotnet/

# Install Python packages
RUN cd /app/backend && pip install -r requirements.txt

# Install Node.js dependencies
RUN cd /app/frontend && npm install

# Build custom metric (Cython extension must be built, not mounted)
WORKDIR /app/backend/api/src/machine_learning/models/model_types/k_nearest_neighbours/custom_metric
RUN python3 setup.py build_ext --inplace

WORKDIR /app

# Restore dotnet packages (dotnet is in /root/.dotnet from install script)
ENV PATH="$PATH:/root/.dotnet"
ENV DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1
RUN cd /app/dotnet && dotnet restore

# Configure git to use host credentials
RUN git config --global --add safe.directory /app

EXPOSE 3000 5000 5001

ENV FLASK_APP=backend/api/api.py
ENV FLASK_ENV=development
ENV PYTHONUNBUFFERED=1
ENV ASPNETCORE_URLS=http://+:5001
ENV ASPNETCORE_ENVIRONMENT=Development

# Keep container alive — services are started via VS Code tasks on folder open
CMD ["sleep", "infinity"]
