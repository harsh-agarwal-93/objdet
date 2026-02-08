FROM ghcr.io/mlflow/mlflow:v2.10.0

# Install curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
