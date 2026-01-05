FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

# mlflow logging needs git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --only-group monitoring
COPY . .
