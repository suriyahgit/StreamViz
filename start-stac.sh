#!/usr/bin/env bash
set -euo pipefail

host="${POSTGRES_HOST:-db}"
port="${POSTGRES_PORT:-5432}"

echo "Waiting for Postgres at ${host}:${port} ..."
until nc -z "${host}" "${port}"; do
  sleep 1
done
echo "Postgres is up."

exec uvicorn stac_fastapi.pgstac.app:app --host 0.0.0.0 --port "${STAC_API_PORT:-8081}"
