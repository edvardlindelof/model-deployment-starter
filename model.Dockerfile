FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app
COPY . .
RUN uv sync --only-group model --frozen

ENTRYPOINT ["uv", "run", "--only-group", "model"]
