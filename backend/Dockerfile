# =============================================================================
# Build stage to create wheel for objdet dependency
# =============================================================================
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app
COPY pyproject.toml uv.lock LICENSE README.md ./
COPY ml/ ml/
RUN uv build

# =============================================================================
# Final stage
# =============================================================================
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Switch to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Copy dependency files
COPY --chown=appuser:appuser pyproject.toml uv.lock LICENSE README.md ./

# Install dependencies (only dependencies, not the project itself)
RUN uv sync --frozen --no-install-project

# Set up virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Copy wheel from builder
COPY --from=builder --chown=appuser:appuser /app/dist /app/dist

# Install the package from wheel
RUN uv pip install /app/dist/*.whl

# Copy application code into backend subdirectory
COPY --chown=appuser:appuser backend/ backend/

EXPOSE 8000

ENV PYTHONPATH=/app

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
