# `auth.py` Review

## Correctness and security blockers

### Critical — authorization fails open when denial auditing fails
**Evidence:** `auth.py:8-11` wraps `audit_denial(user_id, supplied_token)` in `except Exception` and returns `True` from that handler.

An invalid or mismatched token is therefore authorized whenever the audit call raises—for example, due to a logging, output, or future audit-backend failure. An attacker who can trigger that failure, or an ordinary outage in the audit path, bypasses authentication entirely. Denial auditing must not change an authorization decision: preserve `False` for a mismatched token and handle/report audit failures separately.

### High — audit output discloses supplied bearer tokens
**Evidence:** `auth.py:15-16` prints `token` directly in `f"denied user={user_id} token={token}"`; that value is `supplied_token` passed from `authorize` at `auth.py:9`.

Logs and console output commonly have broader access and longer retention than credentials. Recording the full supplied token exposes a reusable secret to anyone who can read that output. Do not log the token value; log a non-secret correlation identifier or a carefully designed, non-reversible fingerprint only if operationally required.

### High — token comparison is not constant-time
**Evidence:** `auth.py:5` compares `supplied_token` and `stored_token` with `==`.

For authentication secrets, ordinary equality comparison can reveal timing differences based on matching prefixes. A remote timing attack may use those differences to help guess a token. Compare same-format secret values with a constant-time primitive (for example, `hmac.compare_digest`) after applying the system’s established token storage/validation approach.

## Maintainability and operational concerns

### Medium — global token cache retains credentials without bounds or lifecycle control
**Evidence:** `auth.py:1` creates the process-global `TOKEN_CACHE`, and `auth.py:6` stores each successful caller’s raw `supplied_token` under `user_id`.

The cache holds plaintext credentials indefinitely, has no size limit, expiry, revocation handling, or synchronization policy, and is not used by this module’s authorization logic. It increases secret exposure in memory and can grow for every distinct user ID. Remove it if unnecessary; otherwise use an explicitly managed cache that avoids raw token storage and defines expiry, invalidation, concurrency, and memory bounds.

### Low — public values and cache contract lack type definitions
**Evidence:** `auth.py:1` defines an untyped global cache, while `auth.py:4` exposes untyped `authorize` parameters and return behavior.

The expected shapes of `user_id`, tokens, scopes, and the cache are implicit, making accidental misuse harder to detect and testing less clear. Add appropriate type annotations and document the authorization contract once the security issues above are resolved.

## Optional observation

`authorize` returns only whether the literal string `"admin"` is in `scopes` (`auth.py:7`). Confirm this intentionally implements the full permission policy; otherwise non-admin permissions are silently denied even with a valid token.
