FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --only-group app --frozen
COPY . .

ENTRYPOINT ["uv", "run", "--only-group", "app"]
