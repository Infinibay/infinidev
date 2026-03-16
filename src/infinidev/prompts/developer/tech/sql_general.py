"""General SQL coding guidelines for the Developer agent."""


def get_prompt() -> str:
    return """\
## SQL Guidelines

### Naming Conventions
- `UPPER CASE` for SQL keywords (`SELECT`, `INSERT`, `WHERE`, `JOIN`).
- `snake_case` for table and column names.
- Pick singular or plural table names and enforce the convention project-wide.
- Name junction/association tables by combining both entity names (e.g. `user_role`, `order_item`).
- Prefix boolean columns with `is_` or `has_` (e.g. `is_active`, `has_paid`).

### Error Handling
- Always handle constraint violation errors (unique, foreign key, check) at the application layer.
- Wrap multi-statement operations in explicit transactions (`BEGIN` / `COMMIT` / `ROLLBACK`).
- Check affected row counts after `UPDATE` and `DELETE` to verify the operation matched expected rows.

### Security
- **Always** use parameterised queries or prepared statements — never interpolate user input into SQL strings.
- Grant minimum necessary privileges to application database users.
- Avoid `SELECT *` in production queries — enumerate only needed columns.
- Use row-level security or application-layer filtering for multi-tenant data.

### Useful Patterns
- Use `EXPLAIN` / `EXPLAIN ANALYZE` to review query plans before deploying against large tables.
- Add indexes on foreign key columns and columns frequently used in `WHERE` / `ORDER BY` clauses.
- Use `COALESCE(column, default)` for nullable column defaults in queries.
- Use Common Table Expressions (CTEs) for readability in complex queries.
- Use `EXISTS` instead of `IN` for correlated subqueries.

### Anti-Patterns to Avoid
- `SELECT *` in application code — it breaks when schema changes and transfers unnecessary data.
- `NOT IN` with nullable columns — use `NOT EXISTS` instead (NULL causes unexpected results).
- Correlated subqueries in loops — rewrite as JOINs or CTEs.
- Missing `WHERE` clause on `UPDATE` or `DELETE` — always double-check filtering conditions.
- Implicit type coercions — cast explicitly to avoid index bypass and subtle bugs.\
"""
