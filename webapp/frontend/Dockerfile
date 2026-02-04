FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen

# Copy application code into frontend subdirectory
COPY . frontend/

EXPOSE 8501

ENV PYTHONPATH=/app

CMD ["uv", "run", "streamlit", "run", "frontend/app.py", "--server.headless", "true"]
