"""OAuth credentials for the ChatGPT (Codex) subscription.

The ``openai_subscription`` provider bills against the user's ChatGPT plan
instead of a metered API key, and the plan is only reachable with an OAuth
access token — ``sk-...`` keys do not open that door.  Rather than run its
own browser + PKCE + loopback-callback dance, Infinidev **reuses the session
the Codex CLI already established**: ``codex login`` writes
``~/.codex/auth.json``, and this module reads, refreshes and writes back that
same file.

Reuse rather than a second login is a deliberate call:

- One session per machine.  Two independent logins to the same account is
  not a security win, it is two sets of long-lived credentials to keep
  correct, and the second one would sit in a file Infinidev invented.
- No new secret custody.  Infinidev never stores a token of its own; the
  file it touches is the one the official client already owns, at its own
  permissions.
- The refresh token is shared state, and that is the whole reason writing
  back is mandatory.  OpenAI rotates refresh tokens: the response to a
  refresh carries a *new* one and retires the old.  A refresh that kept the
  result in memory would leave a dead token in ``auth.json`` and the next
  ``codex`` invocation would demand a fresh login — Infinidev would have
  silently logged the user out of another tool.  So every refresh is
  persisted, atomically, under an advisory lock, re-reading the file first
  in case the CLI refreshed it meanwhile.

Nothing here verifies JWT signatures.  The tokens are read to find out
*when they expire* and *which account they belong to*; the server is the one
that authenticates them, and a client that trusted its own signature check
would gain nothing.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The Codex CLI's own OAuth client id.  Tokens are minted *for* a client, so
# a refresh has to present the same one that issued them — this is not a
# credential, it is the public half of the pair, and using another value
# against a Codex-issued refresh token simply fails.
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"

_TOKEN_URL = "https://auth.openai.com/oauth/token"

# The CLI honours this override, so anything that works for `codex` (a
# proxy, an enterprise gateway) keeps working here.
_TOKEN_URL_ENV = "CODEX_REFRESH_TOKEN_URL_OVERRIDE"

# OpenAI namespaces the subscription claims inside the JWT payload.
_AUTH_CLAIM_NS = "https://api.openai.com/auth"

# Refresh this long before the stated expiry.  A token that dies mid-run
# costs a failed turn, and a run can sit inside one LLM call for minutes.
_REFRESH_MARGIN_SECONDS = 600

_HTTP_TIMEOUT = 30

# Serialises refreshes inside this process; the file lock covers the rest.
_LOCK = threading.Lock()

# One id for the lifetime of the process, mirroring how the CLI tags a
# session.  Generated lazily so importing this module stays free.
_SESSION_ID: str | None = None


class CodexAuthError(RuntimeError):
    """Raised when no usable subscription credential can be produced.

    The message is meant to be shown to the user verbatim — every raise site
    says what to run, because "auth failed" with no next step is the worst
    possible outcome for a credential problem.
    """


@dataclass(frozen=True)
class CodexCredentials:
    """A snapshot of ``auth.json``, parsed."""

    access_token: str
    refresh_token: str
    id_token: str
    account_id: str
    expires_at: float  # epoch seconds; 0.0 when the token carries no exp
    plan_type: str
    path: Path

    def expires_in(self) -> float:
        """Seconds until expiry.  ``inf`` when the token states no expiry."""
        if not self.expires_at:
            return float("inf")
        return self.expires_at - time.time()

    def needs_refresh(self, margin: float = _REFRESH_MARGIN_SECONDS) -> bool:
        return self.expires_in() <= margin


# ── Locating the credentials ─────────────────────────────────────────


def codex_home() -> Path:
    """The Codex CLI's state directory, honouring ``CODEX_HOME``."""
    override = os.environ.get("CODEX_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".codex"


def auth_path() -> Path:
    return codex_home() / "auth.json"


def is_configured() -> bool:
    """Whether a ChatGPT-mode login exists.  Never raises, never refreshes."""
    try:
        return _parse(_read_auth_file(auth_path()), auth_path()) is not None
    except Exception:
        return False


# ── Reading and parsing ──────────────────────────────────────────────


def _read_auth_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise CodexAuthError(
            f"No ChatGPT subscription login found at {path}.\n"
            "Install the Codex CLI and run `codex login` to sign in with your "
            "ChatGPT account, then retry."
        )
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CodexAuthError(
            f"Could not read the Codex credentials at {path}: {exc}.\n"
            "Run `codex login` to regenerate them."
        ) from exc


def _decode_jwt_claims(token: str) -> dict[str, Any]:
    """The payload of a JWT, unverified.  ``{}`` when it isn't one.

    Signature verification is the server's job.  This only needs ``exp`` and
    the account claims, and a client that validated the signature it was
    handed would be checking a token against itself.
    """
    parts = token.split(".")
    if len(parts) != 3:
        return {}
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)  # restore stripped base64 padding
    try:
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return {}


def _parse(data: dict[str, Any], path: Path) -> CodexCredentials | None:
    """Build credentials from a parsed ``auth.json``, or None if unusable."""
    tokens = data.get("tokens") or {}
    access = tokens.get("access_token") or ""
    refresh = tokens.get("refresh_token") or ""
    if not refresh:
        # Without a refresh token there is no way back once access expires,
        # so treat it as "not logged in" rather than limp along.
        return None

    claims = _decode_jwt_claims(access)
    auth_claims = claims.get(_AUTH_CLAIM_NS) or {}

    # The account id lives in two places and they can disagree after an
    # account switch; the token's own claim is the one the API will accept.
    account_id = (
        auth_claims.get("chatgpt_account_id")
        or tokens.get("account_id")
        or ""
    )

    return CodexCredentials(
        access_token=access,
        refresh_token=refresh,
        id_token=tokens.get("id_token") or "",
        account_id=account_id,
        expires_at=float(claims.get("exp") or 0.0),
        plan_type=auth_claims.get("chatgpt_plan_type") or "",
        path=path,
    )


def load_credentials() -> CodexCredentials:
    """Parse ``auth.json``.  Does not refresh."""
    path = auth_path()
    data = _read_auth_file(path)

    mode = (data.get("auth_mode") or "").strip().lower()
    if mode and mode != "chatgpt":
        raise CodexAuthError(
            f"The Codex CLI at {path} is signed in with an API key "
            f"(auth_mode={mode!r}), not a ChatGPT subscription.\n"
            "Run `codex login` and choose 'Sign in with ChatGPT', or switch "
            "Infinidev to the plain `openai` provider and set LLM_API_KEY."
        )

    creds = _parse(data, path)
    if creds is None:
        raise CodexAuthError(
            f"The Codex credentials at {path} carry no refresh token.\n"
            "Run `codex login` to sign in again."
        )
    return creds


# ── Refreshing ───────────────────────────────────────────────────────


def _token_url() -> str:
    return os.environ.get(_TOKEN_URL_ENV, "").strip() or _TOKEN_URL


def _request_refresh(refresh_token: str) -> dict[str, Any]:
    """Exchange a refresh token for a new access token."""
    import httpx

    try:
        resp = httpx.post(
            _token_url(),
            json={
                "client_id": CLIENT_ID,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "scope": "openid profile email",
            },
            headers={"Content-Type": "application/json"},
            timeout=_HTTP_TIMEOUT,
        )
    except Exception as exc:
        raise CodexAuthError(
            f"Could not reach {_token_url()} to refresh the ChatGPT token: {exc}"
        ) from exc

    if resp.status_code != 200:
        # 400 invalid_grant is the common one: the refresh token was revoked,
        # rotated by another client, or the session was signed out.
        detail = resp.text[:200].replace("\n", " ")
        raise CodexAuthError(
            f"Refreshing the ChatGPT token failed ({resp.status_code}): {detail}\n"
            "Run `codex login` to sign in again."
        )

    try:
        payload = resp.json()
    except Exception as exc:
        raise CodexAuthError(
            f"The token endpoint returned a non-JSON response: {exc}"
        ) from exc

    if not payload.get("access_token"):
        raise CodexAuthError(
            "The token endpoint returned no access_token. "
            "Run `codex login` to sign in again."
        )
    return payload


def _persist(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Merge a refresh response into ``auth.json`` on disk, atomically.

    Re-reads the file inside the caller's lock so a concurrent ``codex``
    invocation that rewrote unrelated fields does not lose them, then swaps
    the file in one ``os.replace`` so no reader ever sees a half-written
    credential.
    """
    try:
        data = json.loads(path.read_text())
    except Exception:
        data = {}

    tokens = dict(data.get("tokens") or {})
    tokens["access_token"] = payload["access_token"]
    if payload.get("id_token"):
        tokens["id_token"] = payload["id_token"]
    if payload.get("refresh_token"):
        tokens["refresh_token"] = payload["refresh_token"]

    claims = _decode_jwt_claims(payload["access_token"])
    account_id = (claims.get(_AUTH_CLAIM_NS) or {}).get("chatgpt_account_id")
    if account_id:
        tokens["account_id"] = account_id

    data["tokens"] = tokens
    data.setdefault("auth_mode", "chatgpt")
    data["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z", time.gmtime())

    tmp = path.with_name(path.name + ".infinidev.tmp")
    try:
        # Create the temp file 0600 from the start — a credential must never
        # exist world-readable, not even for the microseconds before a chmod.
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2)
            fh.flush()
            # fsync before the rename: os.replace is atomic with respect to
            # readers, not to a power cut, and a half-written credential that
            # survives a reboot is a login the user has to redo.
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise CodexAuthError(
            f"Refreshed the ChatGPT token but could not write it back to "
            f"{path}: {exc}.\nThe Codex CLI may now need `codex login`."
        ) from exc

    return data


class _FileLock:
    """Advisory lock around ``auth.json``, shared with any other writer.

    Best-effort by design: on a platform without ``fcntl``, or a filesystem
    that refuses locks, the refresh still has to happen.  The atomic replace
    in :func:`_persist` keeps the file consistent either way; the lock only
    narrows the window where two processes each burn a refresh token.
    """

    def __init__(self, path: Path):
        self._path = path.with_name(path.name + ".lock")
        self._fd: int | None = None

    def __enter__(self) -> "_FileLock":
        try:
            import fcntl

            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(self._fd, fcntl.LOCK_EX)
        except Exception as exc:  # pragma: no cover - platform dependent
            logger.debug("auth.json lock unavailable, continuing: %s", exc)
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._fd is None:
            return
        try:
            import fcntl

            fcntl.flock(self._fd, fcntl.LOCK_UN)
        except Exception:  # pragma: no cover - platform dependent
            pass
        finally:
            os.close(self._fd)
            self._fd = None


# ── Public API ───────────────────────────────────────────────────────


def resolve(force_refresh: bool = False) -> CodexCredentials:
    """Credentials guaranteed usable now, refreshing them if near expiry.

    Called on every LLM request, so the common path is a file read and a
    clock comparison.  When a refresh *is* needed the file lock is taken and
    the credentials are re-read inside it: another process may have refreshed
    while this one waited, in which case its token is used and the round trip
    is skipped entirely.
    """
    creds = load_credentials()
    if not force_refresh and not creds.needs_refresh():
        return creds

    with _LOCK, _FileLock(creds.path):
        # Double-checked: whoever held the lock may have just done the work.
        fresh = load_credentials()
        if not force_refresh and not fresh.needs_refresh():
            return fresh

        logger.info(
            "Refreshing ChatGPT subscription token (expired: %s)",
            "yes" if fresh.expires_in() <= 0 else "not yet",
        )
        payload = _request_refresh(fresh.refresh_token)
        data = _persist(fresh.path, payload)
        refreshed = _parse(data, fresh.path)
        if refreshed is None:  # pragma: no cover - _persist always writes tokens
            raise CodexAuthError(
                "Refreshed the ChatGPT token but the result was unusable. "
                "Run `codex login` to sign in again."
            )
        return refreshed


def get_access_token(force_refresh: bool = False) -> str:
    """A valid ChatGPT access token."""
    return resolve(force_refresh=force_refresh).access_token


def session_id() -> str:
    """A stable per-process session id for the Codex backend."""
    global _SESSION_ID
    if _SESSION_ID is None:
        _SESSION_ID = str(uuid.uuid4())
    return _SESSION_ID


def request_headers(account_id: str = "", originator: str = "") -> dict[str, str]:
    """The headers the Codex backend expects beside ``Authorization``.

    ``originator`` identifies the calling client to the backend.  It defaults
    to the Codex CLI's own value because the endpoint only accepts clients it
    knows; ``INFINIDEV_CODEX_ORIGINATOR`` overrides it for anyone whose
    account is provisioned for a different one.
    """
    headers = {
        "session_id": session_id(),
        "originator": (
            originator
            or os.environ.get("INFINIDEV_CODEX_ORIGINATOR", "").strip()
            or "codex_cli_rs"
        ),
    }
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id
    return headers


def status() -> dict[str, Any]:
    """Diagnostics for ``/settings`` and error messages.  Never raises.

    Carries no token material: the point is to answer "am I signed in, as
    what, until when", and a status dict tends to end up in a log.
    """
    try:
        creds = load_credentials()
    except CodexAuthError as exc:
        return {
            "configured": False,
            "path": str(auth_path()),
            "error": str(exc),
        }

    remaining = creds.expires_in()
    return {
        "configured": True,
        "path": str(creds.path),
        "plan": creds.plan_type or "unknown",
        "account_id": creds.account_id,
        "expired": remaining <= 0,
        "expires_in_seconds": None if remaining == float("inf") else int(remaining),
    }
