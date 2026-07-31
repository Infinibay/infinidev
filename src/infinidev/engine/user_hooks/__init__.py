"""User-configured hooks: shell commands the run pauses to consult.

Declared in ``.infinidev/hooks.json``, bound to one of six lifecycle
events, and contributing text — never control. See :mod:`.events` for
what the six are and why the end-of-step and end-of-task events come in
pairs, and :mod:`.config` for the file format.

Nothing in Infinidev ships a default hook. The feature is inert until a
user writes the file.
"""

from infinidev.engine.user_hooks.config import (
    HookSpec,
    get_hooks,
    has_hooks,
    invalidate_cache,
    load_hooks_config,
)
from infinidev.engine.user_hooks.events import UserHookEvent
from infinidev.engine.user_hooks.payload import step_payload, task_payload
from infinidev.engine.user_hooks.render import (
    context_block,
    step_instruction,
    task_instruction,
)
from infinidev.engine.user_hooks.runner import HookOutput, run_hooks

__all__ = [
    "HookOutput",
    "HookSpec",
    "UserHookEvent",
    "context_block",
    "get_hooks",
    "has_hooks",
    "invalidate_cache",
    "load_hooks_config",
    "run_hooks",
    "step_instruction",
    "step_payload",
    "task_instruction",
    "task_payload",
]
