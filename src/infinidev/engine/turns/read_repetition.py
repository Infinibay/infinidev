"""Bound consecutive principal batches that only revisit local read evidence."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from infinidev.engine.engine_logging import extract_tool_error
from infinidev.tools.base.tool_effects import ToolEffects, local_effects_for_name

_WORKSPACE_READ = ToolEffects(reads_workspace=True)
_MAX_RECENT_READS = 256
_WARNING_AFTER_REPEATED_BATCHES = 2
_STOP_AFTER_REPEATED_BATCHES = 4


def read_result_signature(
    name: str, arguments: dict[str, Any], effects: object, body: str, body_sha256: str,
) -> str | None:
    """Identify a successful local read by its call and observed output."""
    if not isinstance(effects, ToolEffects) or effects.is_empty:
        effects = local_effects_for_name(name)
    if effects != _WORKSPACE_READ or not body.strip() or extract_tool_error(body):
        return None
    # Distinct searches can legitimately return the same empty result.
    try:
        identity = json.dumps([name, arguments, body_sha256], sort_keys=True).encode()
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(identity).hexdigest()


class ReadRepetitionGuard:
    """Reset on new evidence, user guidance, or a batch with other effects."""

    def __init__(self) -> None:
        self._seen: dict[str, None] = {}
        self._repeated_batches = 0

    @property
    def warn(self) -> bool:
        return self._repeated_batches >= _WARNING_AFTER_REPEATED_BATCHES

    @property
    def exhausted(self) -> bool:
        return self._repeated_batches >= _STOP_AFTER_REPEATED_BATCHES

    def reset(self) -> None:
        self._seen.clear()
        self._repeated_batches = 0

    def observe_batch(self, reads: list[str] | None) -> None:
        """Observe once after dispatch, so mixed batches can finish their work."""
        if reads is None:
            self.reset()
        elif reads:
            if all(signature in self._seen for signature in reads):
                self._repeated_batches += 1
            else:
                self._repeated_batches = 0
            for signature in reads:
                self._seen[signature] = None
                if len(self._seen) > _MAX_RECENT_READS:
                    self._seen.pop(next(iter(self._seen)))
