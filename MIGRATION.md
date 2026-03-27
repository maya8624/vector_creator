# PostgreSQL To Docker Migration

This project currently uses PostgreSQL via `POSTGRES_URL` in `.env`. The Docker setup below keeps the same database name and credentials, but moves PostgreSQL into a container with `pgvector` enabled.

## Current Settings Found In This Repo

- Connection setting: `POSTGRES_URL`
- Current host before Docker cutover: `localhost:5432`
- Current database: `real_estate_db`
- Current user: `pg-aibot`
- Python config file: `app/core/config.py`
- Vector store setup: `app/database/pgvector_store.py`

## Docker Database

This repo includes `docker-compose.yml` with:

- image: `pgvector/pgvector:pg16`
- host port from `.env.docker`
- database from `.env.docker`
- user from `.env.docker`
- persistent Docker volume: `postgres_data`

Create a local Docker env file first:

```powershell
Copy-Item .env.docker.example .env.docker
```

Then edit `.env.docker` and set your Docker PostgreSQL password locally.

The container runs the init script at `docker/postgres/init/01-enable-pgvector.sql`, which enables:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## 1. Back Up The Current Local Database

Run this before changing anything:

```powershell
& "C:\Program Files\PostgreSQL\17\bin\pg_dump.exe" -h localhost -p 5432 -U pg-aibot -d real_estate_db -Fc -f .\db\real_estate_db.backup
```

You will be prompted for the current local PostgreSQL password.

## 2. Start The Docker Container

```powershell
docker compose --env-file .env.docker up -d
```

Check that the container is healthy:

```powershell
docker compose --env-file .env.docker ps
docker compose --env-file .env.docker logs postgres
```

## 3. Restore The Backup Into Docker

Restore into the Docker PostgreSQL instance on port `5433`:

```powershell
& "C:\Program Files\PostgreSQL\17\bin\pg_restore.exe" -h localhost -p 5433 -U pg-aibot -d real_estate_db --clean --if-exists --no-owner --no-privileges .\db\real_estate_db.backup
```

Use the same user, database, and port values you set in `.env.docker`. You will be prompted for the Docker PostgreSQL password.

## 4. Verify pgvector Is Enabled

Verify the extension exists:

```powershell
& "C:\Program Files\PostgreSQL\17\bin\psql.exe" -h localhost -p 5433 -U pg-aibot -d real_estate_db -c "SELECT extname FROM pg_extension WHERE extname = 'vector';"
```

If needed, enable it manually:

```powershell
& "C:\Program Files\PostgreSQL\17\bin\psql.exe" -h localhost -p 5433 -U pg-aibot -d real_estate_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## 5. Verify The App Connection

After updating `.env`, the app will use:

```text
postgresql://pg-aibot:<docker-password>@localhost:5433/real_estate_db
```

That is the only connection change required for the current Python service.

Your Docker-side settings live in local-only `.env.docker`, which is ignored by git.

## 6. Useful Docker Commands

```powershell
docker compose --env-file .env.docker stop
docker compose --env-file .env.docker start
docker compose --env-file .env.docker down
```

To remove the container and its persisted PostgreSQL data volume, use:

```powershell
docker compose --env-file .env.docker down -v
```

Do that only if you intentionally want a full reset.
