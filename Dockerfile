# syntax=docker/dockerfile:1
# Application image for the Domain Mapper engine (PF-05).
# Builds the project with uv; the Typer CLI arrives in its own Phase 0 row, so
# the container stays idle until then.
FROM python:3.12-slim

# uv, pinned to the version that generated uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Dependencies before sources so this layer caches until the lock file changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY core ./core
COPY adapters ./adapters
RUN uv sync --frozen --no-dev

# No entrypoint yet: the CLI row owns the real command.
CMD ["sleep", "infinity"]
