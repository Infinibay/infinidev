# Review of `auth.py`

## Correctness and security blockers

### **Critical — authentication fails open when denial auditing raises**

`authorize` catches every `Exception` from `audit_denial` and returns `True` (lines 8–11). A malformed logger, an unexpected runtime failure, or another audit error therefore authorizes an invalid token instead of denying it. This is a direct authorization bypass and is the highest-priority correctness/security issue. Denial handling must remain fail-closed; audit failures should be handled separately from the authorization decision and should not convert rejection into approval.

### **High — raw credentials are retained in a process-global cache**

On a successful comparison, line 6 stores `supplied_token`—the credential itself—in the module-global `TOKEN_CACHE` (line 1), keyed by `user_id`. This creates a plaintext secret/credential exposure risk: any code with access to the process or module state can retrieve it, and the cache has no visible expiration, revocation, size bound, or cleanup. The implementation should avoid caching the token; if caching is required, use an appropriate short-lived representation and explicit lifecycle controls.

### **High — raw supplied tokens are printed to the audit log**

`audit_denial` interpolates the unredacted `token` into output on lines 15–16. A rejected credential can consequently be exposed through logs, terminal output, centralized log collection, or retained log files. Tokens are secrets and must not be logged; record a non-sensitive identifier or a carefully designed redacted/fingerprinted value instead. This also makes the fail-open path especially dangerous because the attempted credential is exposed while authorization is incorrectly granted.

### **High — direct token equality may leak timing information**

Line 5 compares `supplied_token` and `stored_token` with ordinary equality. For secret tokens, a constant-time comparison is the safer primitive where an attacker can observe response timing. At minimum, the review should treat this as a token-comparison concern: use a suitable constant-time comparison over consistently represented values, and ensure the stored credential format supports secure verification rather than plaintext comparison.

## Maintainability concerns

### **Medium — untyped, undocumented authorization boundary**

`authorize` and `audit_denial` have no type hints or docstrings (lines 4 and 15). The accepted types and expected behavior of `user_id`, token values, and `scopes` are therefore implicit. Adding types and documentation would clarify the security-sensitive contract, including what happens on invalid input and what the boolean return value means.

### **Medium — mutable global cache makes behavior hard to reason about**

`TOKEN_CACHE = {}` at line 1 is mutable module-global state, and line 6 mutates it. This couples callers through shared state, complicates tests, and can create concurrency and lifecycle problems. There is also no explicit cache policy, such as expiration, invalidation, bounded growth, or synchronization. Prefer an explicit dependency or narrowly scoped session/cache abstraction if token state is genuinely needed.

### **Medium — audit output is coupled to authorization and uses `print`**

The authorization function directly depends on `audit_denial` (lines 8–9), while the audit function uses `print` (line 16). This makes audit behavior difficult to configure, test, monitor, or route through structured logging. A logging interface with clear failure behavior would separate observability from the security decision without exposing the token.

## Optional observations

- The valid-token branch returns whether `"admin"` is in `scopes` (line 7), so a valid non-admin token is deliberately rejected by this function. That policy should be documented so callers do not confuse authentication success with admin authorization.
- The cache is populated before the scope decision is returned (lines 6–7). Thus a valid token lacking the `admin` scope is still retained in `TOKEN_CACHE`; this may be unnecessary credential retention and should be checked against the intended cache contract.
- The broad `except Exception` (line 10) also hides the cause of audit failures, reducing diagnosability even apart from the fail-open bypass.
