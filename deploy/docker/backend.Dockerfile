# =============================================================================
# Backend Dockerfile — FastAPI + Celery (lightweight, no ML dependencies)
# =============================================================================
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Copy backend dependency files only
COPY --chown=appuser:appuser backend/pyproject.toml backend/uv.lock ./

# Install dependencies (frozen, no dev)
RUN --mount=type=cache,target=~/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Copy application code
COPY --chown=appuser:appuser backend/ backend/

EXPOSE 8000

ENV PYTHONPATH=/app

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
