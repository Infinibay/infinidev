# Authentication review

## Correctness and security blockers

### 1. **Critical — denial-audit failures fail open and authorize the request**

In `auth.py:11-12`, any exception raised by `audit_denial(user_id, supplied_token)` causes `authorize()` to return `True`:

```python
except Exception:
    return True
```

That is an authentication bypass: a request with a token that does not equal `stored_token` reaches the denial path, and an audit failure converts the denial into an authorization success. The broad `except Exception` also hides programming, I/O, and operational failures, making the bypass difficult to detect. Denial must remain a denial when auditing fails; audit failure should be surfaced or handled separately without changing the authorization result.

### 2. **High — token equality is not a safe secret-comparison primitive**

`auth.py:5` compares tokens with ordinary `==`:

```python
if supplied_token == stored_token:
```

For secrets, this does not provide an explicit constant-time comparison guarantee and can permit timing side channels. The implementation also compares and stores the token as a raw value rather than using a representation intended for secret verification. Use a well-defined token format and a constant-time comparison (for example, after validating/decoding the token as appropriate) and avoid retaining the presented secret.

### 3. **High — missing-token values can authenticate when both sides are `None`**

There is no validation that either token is present or has the expected type/format before the comparison at `auth.py:5`. If a caller supplies `None` while `stored_token` is also `None` (for example, because credentials are missing or unconfigured), `None == None` is true, so the request is treated as authenticated. The function should reject absent or malformed credentials before comparing them and should fail closed when stored authentication configuration is missing.

### 4. **High — the supplied credential is written to a process-global cache**

On successful comparison, `auth.py:6` stores the caller-provided token in the module-global `TOKEN_CACHE` under `user_id`:

```python
TOKEN_CACHE[user_id] = supplied_token
```

This retains a reusable credential in plaintext for the lifetime of the process and makes it available to any code that can access the module. It also creates cross-request state: entries are never expired, invalidated, bounded, or removed, and a reused `user_id` overwrites prior state. The cache is not consulted by `authorize()` in this file, so it provides exposure and lifecycle risk without contributing to the decision. Do not cache raw tokens; if caching is required, define ownership, expiration, revocation, concurrency, and a non-reversible representation.

### 5. **High — denial auditing prints the raw token**

`auth.py:16` includes `token` directly in stdout:

```python
print(f"denied user={user_id} token={token}")
```

This exposes a credential in terminal output and commonly in centralized logs, CI artifacts, or container log retention. A failed authentication attempt can therefore leak the exact value an attacker supplied, and a legitimate token accidentally sent to the denial path would also be exposed. Audit only non-sensitive metadata and use a controlled logging mechanism with appropriate redaction and escaping; do not log the token.

## Maintainability concerns

- `authorize()` has no type hints, contract documentation, or explicit distinction between authentication (token validity) and authorization (scope membership). The return value is a bare boolean, while the function also mutates global state; callers cannot infer these effects from the signature.
- `scopes` is trusted without validation. A string such as `"not-admin"` has substring membership for `"admin"`, so `"admin" in scopes` can return true for an input that is not a collection of scope names. Require and validate a set/list of normalized scope values, or use an explicit scope-checking API.
- `TOKEN_CACHE` is an untyped mutable global. It is not thread-safe as a defined abstraction and has no documented eviction, expiration, revocation, or privacy policy.
- The audit helper uses `print()` rather than an injected or configured logger. This couples security behavior to stdout and makes severity, destination, redaction, and structured fields difficult to control.
- Catching `Exception` in a security decision removes observability. If an audit operation is best-effort, catch only the expected audit exception, record the failure through a separate safe channel, and preserve the original denial.

## Optional observations

- `user_id` and `supplied_token` are interpolated without escaping. If these values can contain newlines or control characters, log records can be forged or corrupted; structured logging with safe encoding avoids this.
- The successful path returns `True` only when the literal value `"admin"` is present in `scopes`; valid non-admin authentication returns `False`. That may be intentional authorization policy, but the API name `authorize` and lack of documentation make the policy easy to misread.
- There are no visible tests in this small repository exercising wrong-token behavior, audit failures, missing credentials, secret handling, scope types, or cache lifecycle. These cases should be covered before relying on the function for access control.
