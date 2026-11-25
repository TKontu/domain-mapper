# Local compose stack

`compose.yaml` runs the development stack: PostgreSQL 16 with the `vector`
extension, an application image built from the repository `Dockerfile`, and an
opt-in MinIO profile. The raw-storage choice (filesystem vs. MinIO) is still an
open decision, so MinIO is never part of the default stack.

## Prerequisites

- Docker Engine with the `docker compose` plugin.
- A local `.env` (gitignored). Copy the tracked placeholder file and set the
  required credentials; compose refuses to start without them:

  ```sh
  cp .env.example .env
  # edit .env: set DM_POSTGRES_PASSWORD (and DM_MINIO_* before using that profile)
  ```

## Start the database

```sh
docker compose up -d db
docker compose ps            # db is healthy once pg_isready succeeds
```

The `vector` extension is created the first time the `pgdata` volume is initialised
(`deploy/postgres/init/01-vector.sql`). Check it:

```sh
docker compose exec db psql -U "${DM_POSTGRES_USER:-domain_mapper}" \
  -d "${DM_POSTGRES_DB:-domain_mapper}" \
  -c "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
```

The query must return one row. Init scripts run only against an empty data
volume; use `down -v` when you need a clean initialisation.

## Application image

```sh
docker compose build app
docker compose run --rm app python -c "import core; print(core.__name__)"
```

The image is `python:3.12-slim` with dependencies installed by `uv sync --frozen`.
The compose service passes `.env` into the container when the file exists
(`env_file` with `required: false`, so a fresh checkout still builds and runs).

## Optional MinIO profile

MinIO starts only when the profile is requested explicitly:

```sh
DM_MINIO_ROOT_USER=... DM_MINIO_ROOT_PASSWORD=... \
  docker compose --profile minio up -d
```

Set `DM_MINIO_IMAGE` to override the pinned default image, and
`DM_MINIO_ROOT_USER` / `DM_MINIO_ROOT_PASSWORD` (no defaults) in `.env`.
Nothing in the default stack depends on MinIO.

## Stop and clean up

```sh
docker compose down          # stop containers, keep volumes
docker compose down -v       # also delete pgdata/miniodata (fresh init next time)
```

Use a dedicated project name for throwaway checks, e.g.
`docker compose -p dm-pf05 up -d db`, and remove it with
`docker compose -p dm-pf05 down -v`.

## Ports and variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `DM_POSTGRES_DB` | `domain_mapper` | database name |
| `DM_POSTGRES_USER` | `domain_mapper` | database role |
| `DM_POSTGRES_PASSWORD` | — (required) | database password |
| `DM_POSTGRES_PORT` | `5432` | host port for the database |
| `DM_MINIO_ROOT_USER` | — (required with profile) | MinIO root user |
| `DM_MINIO_ROOT_PASSWORD` | — (required with profile) | MinIO root password |
| `DM_MINIO_PORT` / `DM_MINIO_CONSOLE_PORT` | `9000` / `9001` | MinIO API / console ports |
| `DM_MINIO_IMAGE` | pinned mirror | MinIO server image override |
