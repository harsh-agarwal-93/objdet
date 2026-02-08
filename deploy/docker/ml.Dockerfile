# Consolidating Training and Serving into one Dockerfile
# Based on: deploy/Dockerfile.train and deploy/Dockerfile.serve

# =============================================================================
# Build stage: Create wheel
# =============================================================================
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app
COPY pyproject.toml uv.lock LICENSE README.md ./
COPY ml/ ml/
RUN uv build

# =============================================================================
# Base stage: CUDA runtime + System Ops
# =============================================================================
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install uv to correct location
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install system dependencies
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python via uv
COPY .python-version .
RUN uv python install

# =============================================================================
# Intermediate: install common dependencies
# =============================================================================
# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

# Copy dependency files
COPY --chown=appuser:appuser pyproject.toml uv.lock LICENSE README.md ./

# =============================================================================
# Stage: Train
# =============================================================================
FROM base AS train

USER appuser

# Install dependencies (frozen) - only production deps
# Note: we might need dev deps for running tests inside container if needed
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --no-install-project

ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Copy wheel and install
COPY --from=builder --chown=appuser:appuser /app/dist /app/dist
RUN uv pip install /app/dist/*.whl

# Copy configs
COPY --chown=appuser:appuser ml/configs/ ml/configs/

ENTRYPOINT ["python", "-m", "objdet"]
CMD ["--help"]

LABEL org.opencontainers.image.title="ObjDet Training"

# =============================================================================
# Stage: Serve
# =============================================================================
FROM base AS serve

USER appuser

# Install dependencies with tensorrt extra
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra tensorrt

ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Copy wheel and install
COPY --from=builder --chown=appuser:appuser /app/dist /app/dist
RUN uv pip install /app/dist/*.whl

# Copy configs
COPY --chown=appuser:appuser ml/configs/ ml/configs/

# Create directory for models
USER root
RUN mkdir -p /app/models && chown appuser:appuser /app/models
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["python", "-m", "objdet"]
CMD ["serve", "--config", "configs/serving/default.yaml"]

LABEL org.opencontainers.image.title="ObjDet Serving"
