FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --only-group app --only-group models --frozen
COPY . .

ENTRYPOINT ["uv", "run", "--only-group", "app", "--only-group", "models"]
