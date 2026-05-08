FROM python:3.12-slim

WORKDIR /app

# System dependencies

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv

RUN pip install --no-cache-dir uv

# Copy dependency manifests first

COPY pyproject.toml uv.lock ./

# Create virtual environment + install dependencies

RUN uv venv && \
    uv sync --frozen --no-dev


# Copy application source

COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
