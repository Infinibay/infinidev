"""Masking for credentials that would otherwise be rendered on screen.

An API key printed into a settings panel does not stay on that screen: it
lands in terminal scrollback, in screenshots pasted into issues, in
asciinema recordings, and in whatever the user's screen-sharing session is
showing at the time. The value is only ever needed by the HTTP client, so
the UI shows a *shape* — enough to recognise which key is configured,
never enough to use it.

    LLM_API_KEY = sk-ant-••••••••7f2a

Recognition matters: with several providers configured, "is this the
Anthropic key or the OpenRouter one?" is a real question, and a row of
undifferentiated asterisks cannot answer it. So the prefix (which is a
public routing token, not the secret) and the last four characters stay.
"""

from __future__ import annotations

import re

MASK_CHAR = "•"

# A field is secret when its NAME says so. Matching on names rather than on
# value shape means a new provider's key is masked the day it is added,
# without anyone remembering to register it.
_SECRET_NAME_RE = re.compile(
    r"(API_KEY|_KEY$|^KEY$|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|PRIVATE)",
    re.IGNORECASE,
)

# Public prefixes worth preserving: they identify the provider, not the
# secret. Ordered longest-first so "sk-ant-" wins over "sk-".
_PUBLIC_PREFIXES = (
    "sk-ant-api",
    "sk-proj-",
    "sk-ant-",
    "ghp_",
    "gho_",
    "github_pat_",
    "xai-",
    "sk-or-",
    "sk-",
    "pk-",
    "AIza",
    "hf_",
)

# Values below this length reveal too much when partially shown, so they
# are masked whole.
_MIN_LENGTH_FOR_TAIL = 12
_TAIL = 4

# Placeholders that are not secrets at all — local backends take a literal
# string where a key would go, and hiding it just confuses the user.
_NON_SECRET_VALUES = {"ollama", "none", "not-needed", "no-key", "local", "dummy"}


def is_secret(name: str) -> bool:
    """Whether a setting name denotes a credential."""
    return bool(name and _SECRET_NAME_RE.search(name))


def mask_secret(value: object, *, keep_tail: int = _TAIL) -> str:
    """Render *value* as a recognisable but unusable shape."""
    text = "" if value is None else str(value)
    if not text:
        return "(not set)"
    if text.lower() in _NON_SECRET_VALUES:
        return text

    prefix = ""
    for candidate in _PUBLIC_PREFIXES:
        if text.startswith(candidate):
            prefix = candidate
            break

    body = text[len(prefix) :]
    if len(body) < _MIN_LENGTH_FOR_TAIL or keep_tail <= 0:
        return f"{prefix}{MASK_CHAR * 8}"
    return f"{prefix}{MASK_CHAR * 8}{body[-keep_tail:]}"


def mask_if_secret(name: str, value: object) -> str:
    """Mask *value* when *name* denotes a credential; otherwise stringify."""
    if is_secret(name):
        return mask_secret(value)
    return "" if value is None else str(value)


# Below this length a "secret" is a placeholder or an empty-ish sentinel;
# redacting such a value would rewrite unrelated text that happens to
# contain it.
_MIN_REDACTABLE = 12


def configured_secrets() -> list[str]:
    """Every credential currently configured, longest first.

    Longest-first matters: an assistant key that merely *starts* with the
    main key must not be half-redacted into a still-usable remainder.
    """
    try:
        from infinidev.config.settings import settings
    except Exception:
        return []
    values: set[str] = set()
    for name in dir(settings):
        if name.startswith("_") or not is_secret(name):
            continue
        try:
            value = getattr(settings, name)
        except Exception:
            continue
        if isinstance(value, str) and len(value) >= _MIN_REDACTABLE:
            values.add(value)
    return sorted(values, key=len, reverse=True)


def redact(text: str) -> str:
    """Replace any configured credential found in *text* with its mask.

    Defence in depth for text nobody audited: provider SDKs put the key in
    the request URL, so an ``Invalid API key`` exception stringifies
    straight into the transcript with the credential inside it.
    """
    if not text:
        return text
    for value in configured_secrets():
        if value in text:
            text = text.replace(value, mask_secret(value))
    return text
