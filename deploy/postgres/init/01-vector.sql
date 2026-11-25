-- Runs once, on first database initialisation: the Postgres entrypoint executes
-- every script in /docker-entrypoint-initdb.d when the data volume is empty.
-- The compose `db` service mounts this directory read-only.
CREATE EXTENSION IF NOT EXISTS vector;
