# Code Review: `auth.py`

## Correctness and security blockers

### 1. Critical — An audit failure bypasses authentication and authorization

**Evidence:** `authorize` reaches the denial path when `supplied_token != stored_token` (`auth.py:5,8-9`). If `audit_denial(...)` raises any `Exception`, the handler returns `True` (`auth.py:9-11`). This grants access even though the token did not match, and it does so without checking whether `"admin"` is present in `scopes`.

Any routine logging failure—such as an unavailable output stream—or an unexpected bug in the audit function therefore turns a rejected credential into an authorization success. This is a fail-open security bypass.

**Recommendation:** Make every failed token comparison return `False`, regardless of whether audit recording succeeds. Handle audit errors separately (for example, report them through a safe fallback) without changing the authorization result. Catch only exceptions that the audit mechanism is expected to raise rather than using a blanket `except Exception`.

### 2. High — Denied credentials are disclosed in plaintext audit output

**Evidence:** The denial path passes the untrusted `supplied_token` to `audit_denial` (`auth.py:9`), which prints the complete value as `token={token}` (`auth.py:15-16`).

A supplied token is a secret credential even when it is invalid for this account: it may be valid elsewhere, mistyped by one character, or captured from another user. Printing it exposes the credential to terminal output and any collected logs.

**Recommendation:** Never log raw tokens. Record non-secret context such as the user identifier and denial reason. If correlation is essential, use a deliberately designed, non-reversible identifier with an appropriate protected key—not the token itself or a plain unsalted hash.

### 3. High — Secret tokens are compared with a non-constant-time operation

**Evidence:** `auth.py:5` compares `supplied_token` and `stored_token` with ordinary `==`. String/byte equality is not a constant-time credential comparison, so timing can vary based on the compared values and runtime behavior.

Where an attacker can collect sufficiently precise measurements, this can leak information about a stored secret. The direct comparison also indicates that the API expects the plaintext stored token to be available in memory.

**Recommendation:** Verify tokens using the storage scheme appropriate to how they are generated. For high-entropy opaque tokens, store a cryptographic digest and compare fixed-format digest bytes with a constant-time primitive such as `hmac.compare_digest`. If plaintext comparison is unavoidable at this boundary, normalize both operands to the same bytes type and use a constant-time comparison; do not silently coerce mixed types.

### 4. High — Valid credentials are retained in an unbounded module-global cache

**Evidence:** `TOKEN_CACHE` is a mutable module-level dictionary (`auth.py:1`). After a token matches, `authorize` stores the caller-supplied plaintext token under `user_id` (`auth.py:5-6`) before checking whether the caller has the admin scope (`auth.py:7`). Thus even a correctly authenticated non-admin request leaves its credential in global process memory. There is no expiry, size bound, deletion, or consumer in this repository.

This unnecessarily extends secret lifetime, makes credentials accessible to unrelated code that can import the module, and permits unbounded growth across user IDs. Because the cache is populated before authorization succeeds, it also does not represent a cache of successful authorizations.

**Recommendation:** Remove the cache unless a demonstrated caller requires it. If caching is required, do not retain plaintext credentials; define exactly what non-secret result is cached, populate it only after the intended authorization succeeds, and add bounded lifetime, eviction, concurrency behavior, and invalidation on token rotation or revocation.

## Maintainability concerns

### 5. Medium — Authentication, authorization, caching, and audit I/O are coupled

**Evidence:** One function compares credentials (`auth.py:5`), mutates global state (`auth.py:6`), decides an admin scope (`auth.py:7`), and invokes audit output (`auth.py:8-10`). The audit implementation itself writes directly with `print` (`auth.py:16`).

This coupling caused the most serious defect above: an auxiliary audit failure controls the access decision. It also makes the function difficult to test in isolation and leaves callers unable to supply a structured logger or cache policy.

**Recommendation:** Keep the decision path explicit: compute a denial result independently of best-effort auditing, and inject or otherwise isolate audit handling. Introduce a cache abstraction only if the cache has a confirmed requirement; otherwise remove that side effect.

### 6. Medium — The API contract and accepted types are unspecified

**Evidence:** `authorize` and `audit_denial` have no type annotations or docstrings (`auth.py:4,15`). The implementation assumes that `scopes` supports membership testing (`auth.py:7`), that `user_id` is hashable (`auth.py:6`), and that token operands have compatible comparison semantics (`auth.py:5`), but none of these requirements are stated.

Malformed inputs can therefore fail at surprising points. For example, a matching token can be inserted into the global cache before an invalid `scopes` value raises during the membership check (`auth.py:6-7`), leaving a partial side effect.

**Recommendation:** Define and annotate the accepted identifier, token, and scope types and document the boolean return contract. Validate inputs before mutating state. Prefer an immutable scope collection at the boundary if callers do not need mutation.

## Optional observations

- **Scope semantics are exact and case-sensitive.** `auth.py:7` recognizes only the literal string `"admin"`. This is reasonable if it is the documented policy; otherwise, centralize scope names to avoid spelling drift. Do not normalize case unless the product’s scope contract explicitly requires it.
- **No behavioral tests or local callers are present.** The repository contains no tests that execute `authorize`. Add focused tests for matching and mismatching tokens, admin and non-admin scopes, audit failures that must still deny, absence of raw credentials in audit output, and any eventual cache lifecycle. Security-sensitive denial paths should be regression-tested explicitly.
