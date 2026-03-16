"""Redis-specific coding guidelines for the Developer agent."""


def get_prompt() -> str:
    return """\
## Redis Guidelines

### Naming Conventions
- Use `:` as a namespace separator (e.g. `app:user:42:session`).
- Include a version prefix when the key schema changes (e.g. `v2:cache:products`).
- Document all key patterns in a `KEYS.md` or equivalent reference document.
- Keep key names short but descriptive — every byte counts at scale.

### Error Handling
- Always handle connection errors and timeouts — Redis is a network service.
- Use retry logic with exponential backoff for transient failures.
- Never assume a key exists — use `EXISTS` or handle `nil` returns gracefully.
- Set client-side timeouts to avoid blocking indefinitely on network issues.

### Security
- Never store plaintext passwords or PII without encryption.
- Use `AUTH` and TLS in production — never expose Redis without authentication.
- Disable dangerous commands (`FLUSHALL`, `FLUSHDB`, `CONFIG`, `DEBUG`) via `rename-command` in `redis.conf`.
- Use ACLs (Redis 6+) for per-user command and key permissions.
- Bind to `127.0.0.1` or a private network interface — never expose Redis to the public internet.

### Data Structures
- `STRING` for simple key-value caching and counters.
- `HASH` for objects with multiple fields (e.g. user profiles).
- `SORTED SET` for leaderboards, priority queues, and time-series indexes.
- `SET` for unique membership checks and set operations.
- `LIST` for queues (FIFO with `LPUSH` / `RPOP`) and stacks.
- `STREAM` for append-only event logs with consumer groups.

### Useful Patterns
- Always set a `TTL` on cache keys — use `SETEX` or `SET ... EX` to avoid stale data accumulation.
- Use `MULTI` / `EXEC` transactions or Lua scripts for atomic multi-key operations.
- Use `SCAN` (not `KEYS`) for iterating over keys in production — `KEYS` blocks the server.
- Use `OBJECT ENCODING` to verify memory-efficient data representations.
- Use pipelining to batch multiple commands and reduce round trips.

### Anti-Patterns to Avoid
- `KEYS *` in production — it scans the entire keyspace and blocks the server.
- Storing large blobs (>1 MB) without compression — use external storage for large objects.
- Using Redis as a primary database — it is an in-memory store; data loss is possible without persistence.
- Unbounded `LRANGE` / `SMEMBERS` on large collections — always paginate.
- Session keys without a TTL — leads to unbounded memory growth.\
"""
