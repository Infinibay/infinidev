"""Redaction rules applied before event payloads are persisted or indexed.

The event log must stay queryable without becoming a secret dump
(docs/GRAPH_ENGINE_BETA_DESIGN.md §10.3). These rules are deliberately
conservative — they mask high-confidence credential shapes and leave
everything else untouched, because over-redaction silently destroys the
causal record the log exists to keep.

Embeddings are an index, not the source of truth: redaction here applies
to the persisted row itself, so every downstream index inherits it.
"""

from __future__ import annotations

import re

REDACTED = "[REDACTED]"

# key=value / key: value pairs where the key names a credential.
_SECRET_KEY_VALUE = re.compile(
    r"(?i)\b(api[_-]?key|apikey|access[_-]?token|auth[_-]?token|token|"
    r"secret|password|passwd|client[_-]?secret|private[_-]?key|credentials?)"
    r"(\s*[:=]\s*)"
    r"(\"[^\"]+\"|'[^']+'|[^\s\"',;}{)\]]+)"
)

# Authorization: Bearer <token>
_BEARER = re.compile(
    r"(?i)\b(authorization\s*[:=]\s*bearer\s+)[A-Za-z0-9._~+/=-]{8,}"
)

# Provider key shapes that appear unanchored in text.
_SK_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")
_AWS_KEY = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_GH_TOKEN = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")

# PEM private-key blocks, including the body lines.
_PEM_BLOCK = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"
)

# Passwords embedded in connection URLs (scheme://user:pass@host).
_URL_PASSWORD = re.compile(r"(?i)(\b[a-z][a-z0-9+.-]*://[^:/@\s]+:)([^@\s]+)(@)")


def redact_text(text: str) -> str:
    """Mask credential-shaped substrings in *text*."""
    if not text:
        return text
    text = _PEM_BLOCK.sub(REDACTED, text)
    text = _BEARER.sub(lambda m: m.group(1) + REDACTED, text)
    text = _SECRET_KEY_VALUE.sub(lambda m: m.group(1) + m.group(2) + REDACTED, text)
    text = _URL_PASSWORD.sub(lambda m: m.group(1) + REDACTED + m.group(3), text)
    text = _SK_KEY.sub(REDACTED, text)
    text = _AWS_KEY.sub(REDACTED, text)
    text = _GH_TOKEN.sub(REDACTED, text)
    return text


def redact_payload(payload: object) -> object:
    """Recursively redact every string inside *payload* (dict/list/scalar)."""
    if isinstance(payload, str):
        return redact_text(payload)
    if isinstance(payload, dict):
        return {key: redact_payload(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [redact_payload(item) for item in payload]
    return payload


__all__ = ["REDACTED", "redact_payload", "redact_text"]
