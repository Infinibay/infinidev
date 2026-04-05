"""Full prompt variants — registers the original detailed prompts.

This makes ``full`` a first-class variant in the registry, identical to
the existing constants.  No special-case logic needed in consumers.
"""

from infinidev.prompts.variants import register

# ── Loop ──────────────────────────────────────────────────────────────────

from infinidev.engine.loop.context import CLI_AGENT_IDENTITY, LOOP_PROTOCOL

register("full", "loop.identity", CLI_AGENT_IDENTITY)
register("full", "loop.protocol", LOOP_PROTOCOL)

# ── Flow Identities ──────────────────────────────────────────────────────

from infinidev.prompts.flows.develop import DEVELOP_IDENTITY
from infinidev.prompts.flows.research import RESEARCH_IDENTITY
from infinidev.prompts.flows.document import DOCUMENT_IDENTITY
from infinidev.prompts.flows.sysadmin import SYSADMIN_IDENTITY
from infinidev.prompts.flows.explore import EXPLORE_IDENTITY
from infinidev.prompts.flows.brainstorm import BRAINSTORM_IDENTITY

register("full", "flow.develop.identity", DEVELOP_IDENTITY)
register("full", "flow.research.identity", RESEARCH_IDENTITY)
register("full", "flow.document.identity", DOCUMENT_IDENTITY)
register("full", "flow.sysadmin.identity", SYSADMIN_IDENTITY)
register("full", "flow.explore.identity", EXPLORE_IDENTITY)
register("full", "flow.brainstorm.identity", BRAINSTORM_IDENTITY)

# ── Phase Execute ────────────────────────────────────────────────────────

from infinidev.prompts.phases.execute import (
    BUG_EXECUTE, BUG_EXECUTE_IDENTITY,
    FEATURE_EXECUTE, FEATURE_EXECUTE_IDENTITY,
    REFACTOR_EXECUTE, REFACTOR_EXECUTE_IDENTITY,
    OTHER_EXECUTE, OTHER_EXECUTE_IDENTITY,
)

register("full", "phase.bug.execute", BUG_EXECUTE)
register("full", "phase.bug.execute_identity", BUG_EXECUTE_IDENTITY)
register("full", "phase.feature.execute", FEATURE_EXECUTE)
register("full", "phase.feature.execute_identity", FEATURE_EXECUTE_IDENTITY)
register("full", "phase.refactor.execute", REFACTOR_EXECUTE)
register("full", "phase.refactor.execute_identity", REFACTOR_EXECUTE_IDENTITY)
register("full", "phase.other.execute", OTHER_EXECUTE)
register("full", "phase.other.execute_identity", OTHER_EXECUTE_IDENTITY)

# ── Phase Plan ───────────────────────────────────────────────────────────

from infinidev.prompts.phases.plan import (
    PLANNER_IDENTITY,
    BUG_PLAN, BUG_PLAN_IDENTITY,
    FEATURE_PLAN, FEATURE_PLAN_IDENTITY,
    REFACTOR_PLAN, REFACTOR_PLAN_IDENTITY,
    OTHER_PLAN, OTHER_PLAN_IDENTITY,
)

register("full", "phase.planner.identity", PLANNER_IDENTITY)
register("full", "phase.bug.plan", BUG_PLAN)
register("full", "phase.bug.plan_identity", BUG_PLAN_IDENTITY)
register("full", "phase.feature.plan", FEATURE_PLAN)
register("full", "phase.feature.plan_identity", FEATURE_PLAN_IDENTITY)
register("full", "phase.refactor.plan", REFACTOR_PLAN)
register("full", "phase.refactor.plan_identity", REFACTOR_PLAN_IDENTITY)
register("full", "phase.other.plan", OTHER_PLAN)
register("full", "phase.other.plan_identity", OTHER_PLAN_IDENTITY)

# ── Phase Investigate ────────────────────────────────────────────────────

from infinidev.prompts.phases.investigate import (
    BUG_INVESTIGATE, BUG_INVESTIGATE_IDENTITY,
    FEATURE_INVESTIGATE, FEATURE_INVESTIGATE_IDENTITY,
    REFACTOR_INVESTIGATE, REFACTOR_INVESTIGATE_IDENTITY,
    OTHER_INVESTIGATE, OTHER_INVESTIGATE_IDENTITY,
)

register("full", "phase.bug.investigate", BUG_INVESTIGATE)
register("full", "phase.bug.investigate_identity", BUG_INVESTIGATE_IDENTITY)
register("full", "phase.feature.investigate", FEATURE_INVESTIGATE)
register("full", "phase.feature.investigate_identity", FEATURE_INVESTIGATE_IDENTITY)
register("full", "phase.refactor.investigate", REFACTOR_INVESTIGATE)
register("full", "phase.refactor.investigate_identity", REFACTOR_INVESTIGATE_IDENTITY)
register("full", "phase.other.investigate", OTHER_INVESTIGATE)
register("full", "phase.other.investigate_identity", OTHER_INVESTIGATE_IDENTITY)

# Note: phase.investigate.rules is the shared prefix (_INVESTIGATE_RULES)
# which is already baked into the per-type investigate prompts above.
# The generalized/coding variants have it as a separate entry for flexibility.
