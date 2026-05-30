"""OpenAI authentication helpers.

Two official OpenAI paths are intentionally kept separate:

* OpenAI API usage authenticates with an API key (`OPENAI_API_KEY` or
  an explicit Infinidev setting).
* ChatGPT subscription access for Codex is owned by the official Codex
  CLI login flow (`codex login` / `codex login --device-auth`). Infinidev
  does not read, exchange, or reuse Codex OAuth tokens as API bearer
  tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess


_PLACEHOLDER_KEYS = {"", "ollama", "none", "null", "changeme", "your_api_key_here"}


@dataclass(frozen=True)
class CodexAuthStatus:
    """Non-secret summary of the official Codex CLI auth state."""

    cli_available: bool
    authenticated: bool | None
    source: str
    message: str


def resolve_provider_api_key(provider_id: str, configured_api_key: str = "") -> str:
    """Return the API key for a provider, honoring provider env vars.

    For OpenAI, an explicit non-placeholder setting wins; otherwise we
    defer to `OPENAI_API_KEY`, which is the documented API-key path.
    Non-OpenAI providers keep existing behavior.
    """
    key = (configured_api_key or "").strip()
    if provider_id == "openai":
        if key.lower() not in _PLACEHOLDER_KEYS:
            return key
        return os.environ.get("OPENAI_API_KEY", "").strip()
    if provider_id == "openai_codex":
        return ""
    return key


def codex_login_commands() -> tuple[str, str]:
    """Return official Codex login commands for browser and device auth."""
    return ("codex login", "codex login --device-auth")


def codex_auth_status() -> CodexAuthStatus:
    """Check Infinidev/Codex OAuth auth without reading secret token values."""
    try:
        token = load_codex_oauth_token(refresh_if_needed=False)
        suffix = f" for {token.plan_type}" if token.plan_type else ""
        return CodexAuthStatus(
            cli_available=shutil.which("codex") is not None,
            authenticated=True,
            source=str(token.source),
            message=f"Codex OAuth token is configured{suffix}.",
        )
    except Exception:
        pass

    if shutil.which("codex") is None:
        return CodexAuthStatus(
            cli_available=False,
            authenticated=False,
            source=str(infinidev_codex_auth_file()),
            message="Codex OAuth is not configured. Use OpenAI Codex Subscription in /settings or run `/codex login` in classic CLI.",
        )

    try:
        proc = subprocess.run(
            ["codex", "login", "status"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return CodexAuthStatus(
            cli_available=True,
            authenticated=None,
            source="codex login status",
            message=f"Could not run `codex login status`: {exc}",
        )

    output = "\n".join(part for part in (proc.stdout.strip(), proc.stderr.strip()) if part)
    if proc.returncode == 0:
        return CodexAuthStatus(
            cli_available=True,
            authenticated=True,
            source="codex login status",
            message=output or "Codex CLI reports an active login.",
        )

    auth_file = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "auth.json"
    if auth_file.exists():
        return CodexAuthStatus(
            cli_available=True,
            authenticated=None,
            source=str(auth_file),
            message="Codex auth cache exists, but OAuth was not confirmed. Use /settings or `/codex login` to refresh.",
        )
    return CodexAuthStatus(
        cli_available=True,
        authenticated=False,
        source="codex login status",
        message=output or "Codex CLI is installed but no active OAuth login was reported. Use /settings or `/codex login`.",
    )


# ── ChatGPT Codex OAuth (Infinidev-managed) ─────────────────────────────

from datetime import datetime, timedelta, timezone
import base64
import hashlib
import json
import secrets
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

_CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_CODEX_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
_CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
_CODEX_REDIRECT_URI = "http://localhost:1455/auth/callback"
_CODEX_SCOPE = "openid profile email offline_access"


@dataclass(frozen=True)
class CodexOAuthToken:
    """Bearer auth loaded from Infinidev or official Codex OAuth storage."""

    access_token: str
    refresh_token: str
    account_id: str | None
    plan_type: str | None
    is_fedramp_account: bool
    source: Path


@dataclass(frozen=True)
class CodexOAuthFlow:
    """Pending PKCE OAuth flow details shown in the settings modal."""

    authorization_url: str
    state: str
    pending_file: Path


def infinidev_codex_auth_file() -> Path:
    """Return Infinidev's Codex OAuth cache path."""
    return Path(os.environ.get("INFINIDEV_CODEX_HOME", str(Path.home() / ".infinidev" / "openai_codex"))) / "auth.json"


def infinidev_codex_pending_file() -> Path:
    """Return the pending PKCE OAuth flow path."""
    return infinidev_codex_auth_file().with_name("oauth_pending.json")


def start_codex_oauth_flow() -> CodexOAuthFlow:
    """Create and persist a PKCE OAuth flow for ChatGPT Codex auth."""
    verifier = _b64url(secrets.token_bytes(32))
    challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
    state = secrets.token_hex(16)
    params = {
        "response_type": "code",
        "client_id": _CODEX_OAUTH_CLIENT_ID,
        "redirect_uri": _CODEX_REDIRECT_URI,
        "scope": _CODEX_SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": "codex_cli_rs",
    }
    pending_file = infinidev_codex_pending_file()
    _write_json_private(pending_file, {
        "state": state,
        "verifier": verifier,
        "redirect_uri": _CODEX_REDIRECT_URI,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    return CodexOAuthFlow(
        authorization_url=f"{_CODEX_AUTHORIZE_URL}?{urlencode(params)}",
        state=state,
        pending_file=pending_file,
    )


def complete_codex_oauth_flow(input_value: str) -> CodexOAuthToken:
    """Persist tokens from a pasted OAuth redirect URL, query string, or code."""
    code, state = parse_codex_authorization_input(input_value)
    if not code:
        raise RuntimeError("No authorization code found. Paste the redirect URL or code from the browser.")
    pending_file = infinidev_codex_pending_file()
    try:
        pending = _read_json(pending_file)
    except FileNotFoundError:
        raise RuntimeError(
            "No pending OAuth login found. Start OAuth login from /settings again, "
            "then paste the redirect URL from that same login attempt."
        ) from None
    except Exception as exc:
        raise RuntimeError(f"Could not read pending OAuth login state: {exc}") from exc
    expected_state = str(pending.get("state") or "")
    if state and expected_state and state != expected_state:
        raise RuntimeError("OAuth state mismatch. Start the login flow again from settings.")

    resp = httpx.post(
        _CODEX_TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "client_id": _CODEX_OAUTH_CLIENT_ID,
            "code": code,
            "code_verifier": str(pending.get("verifier") or ""),
            "redirect_uri": str(pending.get("redirect_uri") or _CODEX_REDIRECT_URI),
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30.0,
    )
    if not resp.is_success:
        raise RuntimeError(f"Codex OAuth token exchange failed: HTTP {resp.status_code} {resp.text[:300]}")
    auth = _auth_payload_from_token_response(resp.json())
    auth_file = infinidev_codex_auth_file()
    _write_json_private(auth_file, auth)
    try:
        infinidev_codex_pending_file().unlink()
    except FileNotFoundError:
        pass
    return _token_from_auth(auth, auth_file)


def parse_codex_authorization_input(input_value: str) -> tuple[str | None, str | None]:
    """Parse a redirect URL, query string, ``code#state``, or raw code."""
    value = (input_value or "").strip()
    if not value:
        return None, None
    try:
        parsed = urlparse(value)
        if parsed.scheme and parsed.netloc:
            params = parse_qs(parsed.query)
            return _first(params.get("code")), _first(params.get("state"))
    except Exception:
        pass
    if "#" in value:
        code, state = value.split("#", 1)
        return code or None, state or None
    if "code=" in value:
        params = parse_qs(value.lstrip("?"))
        return _first(params.get("code")), _first(params.get("state"))
    return value, None


def load_codex_oauth_token(refresh_if_needed: bool = True) -> CodexOAuthToken:
    """Load ChatGPT OAuth bearer auth for the Codex backend.

    Infinidev's own cache is preferred. The official Codex CLI file cache is
    accepted as a fallback when available.
    """
    candidates = [infinidev_codex_auth_file(), Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "auth.json"]
    errors: list[str] = []
    for auth_file in candidates:
        try:
            auth = _read_json(auth_file)
            if _resolved_auth_mode(auth) == "apikey":
                raise RuntimeError("API-key auth cache, not ChatGPT OAuth")
            token = _token_from_auth(auth, auth_file)
            if refresh_if_needed and token.refresh_token and _jwt_expires_soon(token.access_token):
                auth = _refresh_codex_oauth_token(auth, token.refresh_token)
                if auth_file == infinidev_codex_auth_file():
                    _write_json_private(auth_file, auth)
                token = _token_from_auth(auth, auth_file)
            return token
        except FileNotFoundError:
            continue
        except Exception as exc:
            errors.append(f"{auth_file}: {exc}")
    detail = "; ".join(errors) if errors else "no OAuth cache found"
    raise RuntimeError(f"Codex OAuth token unavailable ({detail}).")


def codex_oauth_headers() -> dict[str, str]:
    """Return auth headers for ChatGPT Codex backend calls."""
    token = load_codex_oauth_token(refresh_if_needed=True)
    headers = {"Authorization": f"Bearer {token.access_token}"}
    if not token.account_id:
        raise RuntimeError("Codex OAuth token is missing chatgpt_account_id. Start OAuth login again from /settings.")
    headers["chatgpt-account-id"] = token.account_id
    if token.is_fedramp_account:
        headers["X-OpenAI-Fedramp"] = "true"
    return headers


def _auth_payload_from_token_response(data: dict[str, Any]) -> dict[str, Any]:
    access = str(data.get("access_token") or "")
    refresh = str(data.get("refresh_token") or "")
    if not access or not refresh:
        raise RuntimeError("Token response missing access_token or refresh_token.")
    claims = _decode_jwt_payload(access) or {}
    return {
        "auth_mode": "chatgpt",
        "tokens": {
            "id_token": data.get("id_token") or access,
            "access_token": access,
            "refresh_token": refresh,
            "account_id": _claim_auth_value(claims, "chatgpt_account_id"),
        },
        "last_refresh": datetime.now(timezone.utc).isoformat(),
    }


def _token_from_auth(auth: dict[str, Any], source: Path) -> CodexOAuthToken:
    tokens = auth.get("tokens")
    if not isinstance(tokens, dict):
        raise RuntimeError("auth cache does not contain OAuth tokens")
    access = str(tokens.get("access_token") or "").strip()
    refresh = str(tokens.get("refresh_token") or "").strip()
    if not access:
        raise RuntimeError("auth cache is missing access_token")
    claims = _extract_claims(tokens, access)
    account_id = str(tokens.get("account_id") or "").strip() or _claim_auth_value(claims, "chatgpt_account_id")
    plan_type = _claim_auth_value(claims, "chatgpt_plan_type")
    return CodexOAuthToken(
        access_token=access,
        refresh_token=refresh,
        account_id=account_id or None,
        plan_type=str(plan_type) if plan_type else None,
        is_fedramp_account=bool(_claim_auth_value(claims, "chatgpt_account_is_fedramp")),
        source=source,
    )


def _refresh_codex_oauth_token(auth: dict[str, Any], refresh_token: str) -> dict[str, Any]:
    resp = httpx.post(
        os.environ.get("CODEX_REFRESH_TOKEN_URL_OVERRIDE", _CODEX_TOKEN_URL),
        data={"client_id": _CODEX_OAUTH_CLIENT_ID, "grant_type": "refresh_token", "refresh_token": refresh_token},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30.0,
    )
    if resp.status_code == 401:
        raise RuntimeError("refresh token expired or was revoked")
    resp.raise_for_status()
    refreshed = resp.json()
    tokens = dict(auth.get("tokens") or {})
    for src, dst in (("id_token", "id_token"), ("access_token", "access_token"), ("refresh_token", "refresh_token")):
        if refreshed.get(src):
            tokens[dst] = refreshed[src]
    auth = dict(auth)
    auth["tokens"] = tokens
    auth["last_refresh"] = datetime.now(timezone.utc).isoformat()
    return auth


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise RuntimeError(f"unexpected JSON shape in {path}")
    return data


def _write_json_private(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    tmp.replace(path)


def _resolved_auth_mode(auth: dict[str, Any]) -> str:
    mode = str(auth.get("auth_mode") or "").lower().replace("_", "")
    if mode:
        return mode
    if auth.get("OPENAI_API_KEY"):
        return "apikey"
    return "chatgpt"


def _jwt_expires_soon(jwt: str) -> bool:
    exp = _jwt_expiration(jwt)
    return bool(exp and exp <= datetime.now(timezone.utc) + timedelta(minutes=2))


def _jwt_expiration(jwt: str) -> datetime | None:
    claims = _decode_jwt_payload(jwt)
    exp = claims.get("exp") if claims else None
    if not isinstance(exp, (int, float)):
        return None
    return datetime.fromtimestamp(exp, tz=timezone.utc)


def _extract_claims(tokens: dict[str, Any], access_token: str) -> dict[str, Any]:
    id_token = tokens.get("id_token")
    if isinstance(id_token, dict):
        return id_token
    if isinstance(id_token, str):
        claims = _decode_jwt_payload(id_token)
        if claims:
            return claims
    return _decode_jwt_payload(access_token) or {}


def _claim_auth_value(claims: dict[str, Any], key: str) -> Any:
    auth_claims = claims.get("https://api.openai.com/auth")
    if isinstance(auth_claims, dict) and key in auth_claims:
        return auth_claims[key]
    return claims.get(key)


def _decode_jwt_payload(jwt: str) -> dict[str, Any] | None:
    parts = jwt.split(".")
    if len(parts) < 2 or not parts[1]:
        return None
    try:
        raw = base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4))
        data = json.loads(raw)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _first(values: list[str] | None) -> str | None:
    return values[0] if values else None
