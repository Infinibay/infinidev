"""MySQL/MariaDB-specific coding guidelines for the Developer agent."""


def get_prompt() -> str:
    return """\
## MySQL / MariaDB Guidelines

Follow all general SQL guidelines, plus these MySQL-specific practices.

### Types
- Use `BIGINT UNSIGNED AUTO_INCREMENT` for integer primary keys, or `CHAR(36)` for UUID keys.
- Use `DATETIME` for timestamps — handle timezone conversions explicitly in the application layer.
- Use the `JSON` type (MySQL 5.7.8+) for document storage with `JSON_EXTRACT` for queries.
- Use `utf8mb4` charset and `utf8mb4_unicode_ci` collation for full Unicode support.

### Engine
- Always use `InnoDB` — it supports transactions, foreign key constraints, and row-level locking.
- Never use `MyISAM` for new tables.

### Indexing
- Composite index column order matters — place the most selective column first for range queries.
- Use `EXPLAIN FORMAT=JSON` for detailed query plan analysis.
- Use covering indexes (include all `SELECT` columns in the index) to avoid table lookups.
- Keep index key length short — prefer integer keys over long string keys.

### Features
- Use `INSERT ... ON DUPLICATE KEY UPDATE` for upsert operations.
- Use `LOAD DATA INFILE` for efficient bulk imports.
- Use `GENERATED COLUMNS` (stored or virtual) for computed values.
- Use `ROW_NUMBER()` and window functions (MySQL 8.0+) instead of user variables for row numbering.

### Security
- Use prepared statements (PDO, `mysql2` gem, `mysql-connector-python`) — never build queries by string concatenation.
- Disable `LOAD DATA LOCAL INFILE` in production configurations.
- Use `GRANT` with minimal privileges per application database user.
- Set `sql_mode` to include `STRICT_TRANS_TABLES` and `NO_ZERO_DATE`.

### Anti-Patterns to Avoid
- `ENUM` columns — use a lookup/reference table instead for maintainability.
- `utf8` charset (3-byte, incomplete) — always use `utf8mb4`.
- `GROUP BY` without `ONLY_FULL_GROUP_BY` SQL mode — leads to non-deterministic results.
- `LOCK TABLES` in application code — use row-level locking with `SELECT ... FOR UPDATE`.
- Implicit charset/collation mismatches between tables and columns.\
"""
