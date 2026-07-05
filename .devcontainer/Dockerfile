ARG NODE_VERSION_NUMBER=22
ARG PYTHON_VERSION_NUMBER=3.9
ARG PYTHON_IMAGE_VERSION=${PYTHON_VERSION_NUMBER}-slim

FROM python:${PYTHON_IMAGE_VERSION}
ARG NODE_VERSION_NUMBER

WORKDIR /app

# Create directories for backend and frontend
RUN mkdir -p /app/backend /app/frontend

# Copy package files
COPY frontend/package.json /app/frontend/
COPY backend/requirements.txt /app/backend/

# Install all dependencies in one efficient layer
# This includes: build tools, Node.js, npm, Python packages
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        graphviz \
        curl \
        gnupg \
        ca-certificates \
        apt-transport-https && \
    curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION_NUMBER}.x | bash - && \
    apt-get install -y nodejs && \
    cd /app/backend && pip install -r requirements.txt && \
    cd /app/frontend && npm install && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY frontend/ /app/frontend/
COPY backend/ /app/backend/

# Build custom metric
WORKDIR /app/backend/api/src/machine_learning/models/model_types/k_nearest_neighbours/custom_metric
RUN python3 setup.py build_ext --inplace

WORKDIR /app

EXPOSE 5000 5001

ENV FLASK_APP=backend/api/api.py
ENV FLASK_ENV=development
ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "python -m flask run --host=0.0.0.0 --port=5000 --reload"]
