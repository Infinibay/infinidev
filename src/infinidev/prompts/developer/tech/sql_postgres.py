"""PostgreSQL-specific coding guidelines for the Developer agent."""


def get_prompt() -> str:
    return """\
## PostgreSQL Guidelines

Follow all general SQL guidelines, plus these PostgreSQL-specific practices.

### Types
- Use `UUID` with `gen_random_uuid()` for primary keys when appropriate.
- Use `TIMESTAMPTZ` not `TIMESTAMP` — always store timestamps with timezone information.
- Use `JSONB` not `JSON` for queryable JSON data (supports indexing and operators).
- Use `TEXT` not `VARCHAR(n)` unless a length constraint is semantically meaningful.
- Use `BOOLEAN` not integer flags for true/false values.

### Indexing
- Create `GIN` indexes for `JSONB` and array columns.
- Use partial indexes for common filters: `CREATE INDEX idx ON t(col) WHERE deleted_at IS NULL`.
- Use the `pg_trgm` extension with `GIN` indexes for efficient `LIKE` / `ILIKE` searches.
- Use `CONCURRENTLY` when adding indexes to large tables in production.

### Features
- Use `INSERT ... ON CONFLICT DO UPDATE` for upsert operations.
- Use the `RETURNING` clause to get inserted/updated rows without a second query.
- Use `LISTEN` / `NOTIFY` for lightweight event-driven pub-sub.
- Use `COPY` for bulk inserts and exports — significantly faster than row-by-row `INSERT`.
- Use `pg_stat_statements` for query performance analysis.

### Security
- Use Row-Level Security (RLS) policies for multi-tenant data isolation.
- Use `SECURITY DEFINER` functions sparingly — always set `SET search_path = public` explicitly.
- Avoid connecting to the database as a superuser from application code.
- Use `pg_hba.conf` to restrict connection sources.

### Anti-Patterns to Avoid
- `SERIAL` columns — use `GENERATED ALWAYS AS IDENTITY` (PostgreSQL 10+).
- Storing timestamps without timezone (`TIMESTAMP` instead of `TIMESTAMPTZ`).
- Running `VACUUM` from application code — let autovacuum handle it.
- `pg_sleep()` in production queries.
- `SELECT FOR UPDATE` without `NOWAIT` or `SKIP LOCKED` in high-concurrency scenarios.\
"""
