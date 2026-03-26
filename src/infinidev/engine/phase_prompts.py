"""Backward-compatible re-exports from prompts.phases.

All phase prompts, identities, and strategies have moved to
src/infinidev/prompts/phases/ for better organization.
"""

from infinidev.prompts.phases import (  # noqa: F401
    PhaseStrategy,
    STRATEGIES,
    get_strategy,
)
