FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock LICENSE README.md ./

# Copy ML source code for local package installation
COPY ml/src/ ml/src/

# Install dependencies
RUN uv sync --frozen

# Copy application code into backend subdirectory
COPY backend/ backend/

EXPOSE 8000

ENV PYTHONPATH=/app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

CMD ["uv", "run", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
