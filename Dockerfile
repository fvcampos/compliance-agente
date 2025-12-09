FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install uv (multi-stage copy, não precisa de curl)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files first (cache layer)
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Produção: sem -w (watch mode)
CMD ["uv", "run", "chainlit", "run", "src/app/ui.py", "--port", "8000", "--host", "0.0.0.0"]
