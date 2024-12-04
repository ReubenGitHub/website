# Define the Python version as a build argument
ARG NODE_VERSION_NUMER=18
ARG PYTHON_VERSION_NUMER=3.9
ARG PYTHON_IMAGE_VERSION=${PYTHON_VERSION_NUMER}-slim

# Step 1: Build the Node.js application
FROM node:${NODE_VERSION_NUMER} AS frontend-build
# Set working directory for frontend
WORKDIR /frontend
# Copy the package.json and package-lock.json
COPY package.json package-lock.json ./
# Install dependencies for the frontend (React, etc.)
RUN npm install
# Copy the rest of the frontend source code
COPY src/ ./src/
COPY public/ ./public/
# Build the frontend application (adjust if using create-react-app or other tool)
RUN npm run build

# Step 2: Build the Python (Flask) application
FROM python:${PYTHON_IMAGE_VERSION} AS backend-build
# Set working directory for backend
WORKDIR /backend
# Copy the Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# Copy the backend source code (Flask API and related files)
COPY api/ ./api/
COPY src/ ./src/

# Step 3: Final image with both Node.js and Python
FROM python:${PYTHON_IMAGE_VERSION}
ARG PYTHON_VERSION_NUMER
# Install Node.js (since the final container will run both backend and frontend)
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs
# Install global dependencies for running both backend and frontend
# RUN apt-get install -y build-essential
# Set working directories for both frontend and backend
WORKDIR /app
# Copy frontend build from the previous stage
COPY --from=frontend-build /frontend/build/ /app/build/
COPY --from=frontend-build /frontend/package.json /app/package.json
# Copy the backend source code and installed dependencies from the previous stage
COPY --from=backend-build /backend/ /app/
COPY --from=backend-build /usr/local/lib/python${PYTHON_VERSION_NUMER}/site-packages /usr/local/lib/python${PYTHON_VERSION_NUMER}/site-packages
COPY --from=backend-build /usr/local/bin /usr/local/bin
# Expose the necessary ports for Flask API (typically 5000) and frontend (typically 3000)
EXPOSE 5000 3000
# Set environment variables if necessary (optional)
ENV FLASK_APP=api/api.py
# Command to run both the frontend (React) and backend (Flask) concurrently
CMD ["sh", "-c", "python3 -m flask run --host=0.0.0.0 --port=5000"]
