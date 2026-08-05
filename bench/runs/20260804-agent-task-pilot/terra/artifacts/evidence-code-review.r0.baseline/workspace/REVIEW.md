# Authorization Review

## Correctness and Security Blockers

### High — Audit failures grant authorization (fail-open)
**Evidence:** `auth.py:8-10` wraps `audit_denial(user_id, supplied_token)` in a broad `except Exception` and returns `True` if that call raises.

A denied request can therefore be authorized whenever audit logging fails—for example, after a logging sink, output stream, formatter, or test double error. Authorization must not depend on successful denial auditing; this exception path is an authentication/authorization bypass.

### High — Supplied bearer tokens are exposed in audit output
**Evidence:** `auth.py:15-16` passes `token` to the audit function, which prints `denied user={user_id} token={token}`.

`supplied_token` is credential material. Writing it in plaintext to standard output can place it in application logs, centralized log stores, terminal history/capture, and monitoring systems. Anyone who can read those logs may be able to replay a valid token that was rejected for another reason or inspect sensitive attempted credentials.

### High — Token equality comparison is not timing-safe
**Evidence:** `auth.py:5` compares `supplied_token` and `stored_token` with `==`.

For secret tokens, ordinary equality can stop at the first differing character. Its runtime can leak prefix-match information to an attacker able to make repeated measurements, which can help guess a stored token. Use a constant-time comparison designed for secrets after ensuring both inputs have an appropriate normalized representation.

## Maintainability Concerns

### Medium — Raw tokens are retained in a module-global cache without lifecycle controls
**Evidence:** `auth.py:1` defines the mutable process-global `TOKEN_CACHE`, and `auth.py:6` stores each accepted `supplied_token` under `user_id`.

The cache retains plaintext credentials indefinitely, has no documented consumer, eviction, expiration, synchronization, or cleanup behavior, and can grow with every distinct user. This expands the credential-exposure surface and makes behavior dependent on process lifetime and shared mutable state.

### Low — The authorization contract is implicit and lacks types or validation
**Evidence:** `auth.py:4` accepts unannotated `user_id`, `supplied_token`, `stored_token`, and `scopes`; `auth.py:7` assumes `scopes` supports membership testing for the exact string `"admin"`.

Without explicit types and input contracts, callers can supply unexpected values (including `None` or custom containers), causing errors or inconsistent authorization behavior. Document and validate the expected token and scope representations.

## Optional Observations

- **Evidence:** `auth.py:5-7` grants access only when tokens match and the literal `"admin"` is present in `scopes`. This is a narrow admin check rather than a general permission decision; confirm that it matches all intended authorization modes.
- **Evidence:** `auth.py:1,6` caches a token after comparison but `authorize` uses the separately supplied `stored_token` for each decision. Unless another component consumes the cache, this write has no effect on the result and is difficult to justify.
