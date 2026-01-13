# PostgreSQL Migration Guide

## Overview

This document describes the migration from SQLite to a dual-database architecture supporting both SQLite (local development) and PostgreSQL (AWS RDS production).

## Architecture

```
Current:
  Application Code → database.py → SQLite

Target:
  Application Code → database.py (facade) → SQLAlchemy Engine
                                                    ↓
                                         DATABASE_URL env var
                                                    ↓
                              ┌─────────────────────┴─────────────────────┐
                              ↓                                           ↓
                           SQLite                                   PostgreSQL
                        (local dev)                                (AWS RDS)
```

## Files Created/Modified

### New Files

| File | Purpose |
|------|---------|
| `backend/src/config.py` | Added `DatabaseConfig` class with environment detection |
| `backend/src/db/__init__.py` | Package exports |
| `backend/src/db/engine.py` | SQLAlchemy engine singleton with pooling |
| `backend/src/db/models.py` | Table definitions (SQLAlchemy Core) |
| `backend/src/db/repository.py` | Query methods returning DataFrames |
| `backend/migrations/migrate_sqlite_to_pg.py` | One-time data migration script |

### Modified Files

| File | Change |
|------|--------|
| `backend/src/database.py` | Converted to facade delegating to repository |
| `backend/requirements.txt` | Added sqlalchemy, psycopg2-binary |

## Usage

### Local Development (SQLite - Default)

No configuration needed. Uses `backend/data/analytics.db`:

```python
from src.db import AnalyticsRepository

repo = AnalyticsRepository()
df = repo.get_games(sport="NFL")
```

### Production (PostgreSQL)

Set environment variable:

```bash
export DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

### Backward Compatibility

Existing code using the old pattern still works:

```python
from src.database import get_connection, get_games

conn = get_connection()  # Shows deprecation warning
df = get_games(conn, sport="NFL")  # Works (conn is ignored internally)
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | None (uses SQLite) |
| `SQLITE_DB_PATH` | Path to SQLite database | `backend/data/analytics.db` |
| `DB_POOL_SIZE` | Connection pool size | 5 |
| `DB_MAX_OVERFLOW` | Max overflow connections | 10 |
| `DB_POOL_RECYCLE` | Seconds before recycling | 3600 |
| `DB_PRE_PING` | Verify connections before use | true |
| `USE_NULL_POOL` | Use NullPool for RDS Proxy | false |

## Data Migration

### Dry Run (Preview)

```bash
cd backend
PYTHONPATH=.. python -m migrations.migrate_sqlite_to_pg \
    --sqlite-path data/analytics.db \
    --pg-url postgresql://user:pass@host:5432/dbname \
    --dry-run
```

### Actual Migration

```bash
cd backend
PYTHONPATH=.. python -m migrations.migrate_sqlite_to_pg \
    --sqlite-path data/analytics.db \
    --pg-url postgresql://user:pass@host:5432/dbname
```

## AWS RDS Setup

### 1. Create RDS Instance

- Instance class: `db.t3.micro` (Free Tier)
- Engine: PostgreSQL 15+
- Storage: 20 GB gp2
- VPC: Same as Lambda functions
- Security group: Allow inbound from Lambda security group on port 5432

### 2. Create Database

```sql
CREATE DATABASE analytics;
```

### 3. Configure Lambda

Add environment variables:
- `DATABASE_URL`: `postgresql://user:pass@rds-endpoint:5432/analytics`
- `USE_NULL_POOL`: `true` (if using RDS Proxy)

### 4. Update IAM Roles

Lambda execution role needs:
- `rds-db:connect` (for IAM authentication)
- Or use password authentication via Secrets Manager

## Key Technical Decisions

### SQLAlchemy Core vs ORM

Using SQLAlchemy Core (not ORM) because:
- Existing code returns pandas DataFrames
- Core provides database-agnostic SQL
- No ORM overhead
- `pd.read_sql()` works directly with Core queries

### Connection Pooling Strategy

| Scenario | Pool Type | Why |
|----------|-----------|-----|
| SQLite | QueuePool (size=1) | SQLite doesn't benefit from pools |
| PostgreSQL (direct) | QueuePool | Maintains connection pool |
| Lambda + RDS Proxy | NullPool | Let proxy handle pooling |

### Upsert Handling

Different syntax for SQLite vs PostgreSQL handled in repository:

```python
# PostgreSQL
stmt = postgresql.insert(games).on_conflict_do_update(constraint="uq_game", ...)

# SQLite
stmt = sqlite.insert(games).on_conflict_do_update(index_elements=[...], ...)
```

## Remaining Steps

1. **Test with SQLite**: Run the test script to verify imports work
2. **Set up AWS RDS**: Create PostgreSQL instance
3. **Run migration**: Use the migration script
4. **Update Lambda**: Add DATABASE_URL environment variable
5. **Test end-to-end**: Verify Lambda connects to RDS

## Testing

```bash
cd backend
PYTHONPATH=. python3 -c "
from src.db import AnalyticsRepository

repo = AnalyticsRepository()
print(f'Backend: {repo.config.backend.value}')
print(f'Games: {repo.get_game_count()}')
print(f'Sports: {repo.get_sports()}')
"
```

## Troubleshooting

### Import Errors

Ensure `PYTHONPATH` includes the backend directory:
```bash
cd backend
PYTHONPATH=. python3 your_script.py
```

### Connection Refused (Lambda to RDS)

- Check security group allows Lambda's security group
- Verify Lambda is in same VPC as RDS
- Check RDS is publicly accessible if needed

### Connection Pool Exhausted

- Increase `DB_POOL_SIZE` and `DB_MAX_OVERFLOW`
- Use `USE_NULL_POOL=true` with RDS Proxy
- Check for connection leaks (unclosed connections)

## Files Reference

### config.py (DatabaseConfig)

```python
from src.config import DatabaseConfig

config = DatabaseConfig.from_env()
print(config.backend)  # DatabaseBackend.SQLITE or POSTGRESQL
print(config.url)      # Connection URL
```

### db/repository.py (AnalyticsRepository)

```python
from src.db import AnalyticsRepository

repo = AnalyticsRepository()

# Games
df = repo.get_games(sport="NFL", team="Chiefs")
repo.insert_games(df, sport="NFL")
repo.get_all_teams(sport="NFL")
repo.get_sports()
repo.get_date_range(sport="NFL")
repo.get_game_count(sport="NFL")

# Ratings
df = repo.get_ratings(sport="NFL")
repo.insert_ratings(df)
repo.get_latest_ratings(sport="NFL")

# Backtesting (joins games with ratings)
df = repo.get_games_with_ratings(sport="NFL", start_date, end_date)
```

### db/engine.py

```python
from src.db.engine import get_engine, reset_engine

engine = get_engine()  # Singleton, reuses same engine
reset_engine()         # For testing, clears singleton
```

### db/models.py

```python
from src.db.models import games, historical_ratings, create_tables, metadata

# Create tables
create_tables(engine)

# Use tables in queries
from sqlalchemy import select
query = select(games).where(games.c.sport == "NFL")
```
