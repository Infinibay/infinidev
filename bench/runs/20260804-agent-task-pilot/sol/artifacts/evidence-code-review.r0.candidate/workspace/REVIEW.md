# Security and Correctness Review of `auth.py`

## Blockers

### Critical — Authorization fails open when denial auditing raises

**Evidence:** On a token mismatch, `authorize` calls `audit_denial` at `auth.py:9`. The broad `except Exception` at `auth.py:10` then returns `True` at `auth.py:11`. Thus an operational failure in logging/auditing—such as an unavailable output stream or an exception while formatting an unusual value—turns an invalid token into successful authorization.

**Impact:** Authentication can be bypassed precisely when the denial audit path malfunctions. Security decisions must not depend on telemetry succeeding.

**Action:** Fail closed: a mismatched token must always return `False`. Handle audit failures separately without changing the authorization result, and catch only exceptions that the audit implementation can meaningfully handle.

### High — Denied credentials are disclosed verbatim through audit output

**Evidence:** The caller-controlled `supplied_token` is passed into `audit_denial` at `auth.py:9`, interpolated without redaction at `auth.py:16`, and written to standard output. The resulting record has the form `denied user=... token=<supplied credential>`.

**Impact:** Valid tokens entered against the wrong account, mistyped secrets, and attacker-supplied content can be exposed to console captures, centralized logs, support tooling, or anyone with log access. Embedded control characters could also forge or corrupt plain-text log records.

**Action:** Never log the supplied token. Record only non-secret context needed for investigation (for example, a safely encoded user identifier and outcome) through the application's structured logging/audit facility.

### High — Raw tokens are retained globally even when authorization is denied

**Evidence:** After token equality succeeds at `auth.py:5`, the raw `supplied_token` is stored in process-global `TOKEN_CACHE` at `auth.py:6` **before** the admin-scope decision at `auth.py:7`. Consequently, a user with a valid token but without the `admin` scope receives `False` while their credential remains in the global cache.

**Impact:** This unnecessarily extends credential lifetime and increases exposure through memory inspection, debugging, accidental introspection, or unrelated code that can mutate/read the module global. It also makes the cache represent authentication success rather than the function's authorization result, which is easy for later callers to misuse.

**Action:** Do not retain plaintext bearer tokens. If caching is required by a defined contract, store a non-reversible verifier or narrowly scoped authorization result with explicit expiry, and populate it only at the intended successful stage.

## Other security and correctness concerns

### Medium — Secret comparison is not constant-time

**Evidence:** `auth.py:5` compares the supplied and stored tokens with ordinary `==`.

**Impact:** Ordinary equality is not intended to provide constant-time secret comparison. In an environment where an attacker can collect sufficiently precise timing measurements, comparison timing may reveal information about the stored token. Practical exploitability depends on token type, runtime, and deployment boundary, but authentication code should not rely on those conditions.

**Action:** Require byte/string token inputs and compare appropriately encoded values with a constant-time primitive such as `hmac.compare_digest`. This complements—not replaces—strong, random tokens and secure storage.

## Maintainability concerns

### Medium — `TOKEN_CACHE` has undefined ownership and lifecycle

**Evidence:** `TOKEN_CACHE = {}` at `auth.py:1` is a mutable process-global dictionary. There is no size bound, expiry, invalidation on token rotation, synchronization policy, or encapsulated access path.

**Impact:** Long-running processes can retain stale credentials indefinitely and grow the cache without bound. Behavior can diverge across worker processes, and concurrent callers or tests share mutable state implicitly.

**Action:** First define whether caching is actually required. If it is, place it behind an explicit cache abstraction with bounded size, expiry and invalidation rules, concurrency guarantees, and tests; otherwise remove it.

### Low — The callable contract is implicit and accepts structurally invalid inputs

**Evidence:** `authorize` has no type annotations or documented input contract (`auth.py:4`). It assumes token values are comparable at `auth.py:5` and that `scopes` supports membership testing at `auth.py:7`. For example, `scopes=None` raises instead of producing a deliberate denial.

**Impact:** Misconfigured callers can trigger inconsistent runtime failures, and unusual equality or membership implementations can execute user-defined behavior inside a security-sensitive decision.

**Action:** Add explicit types and validate or normalize inputs at the trust boundary. Invalid token or scope inputs should produce a deliberate fail-closed result, with tests covering empty, null, and malformed values.

## Optional observations

- `audit_denial` uses `print` (`auth.py:16`) rather than a structured audit interface. After removing all credentials from the event, structured logging would make destination, encoding, severity, and failure handling explicit.
- The function combines token authentication, admin authorization, credential caching, and auditing in one path (`auth.py:4-12`). Keeping the authorization decision small and isolating non-security side effects would make fail-closed behavior easier to reason about and test.
- Focused tests should lock in the critical invariants: a mismatched token is denied even if auditing raises; no audit record contains `supplied_token`; a denied request does not retain credentials; and only a valid token with the required scope is authorized.
