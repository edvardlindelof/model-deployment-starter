FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app
COPY . .
RUN uv sync --only-group app --frozen

ENTRYPOINT ["uv", "run", "--only-group", "app"]
