"""Shared semantic vocabulary for the stage and task planners."""

from __future__ import annotations


PLANNING_VOCABULARY = """\
## Planning vocabulary

A **Goal** is the user-owned outcome. Its literal requirements, exclusions and \
authorization remain stable while planning tactics change.

A **Stage** is the next result whose evidence can change what should be planned \
after it. A Stage has an outcome and observations that decide whether that \
outcome was reached.

A **Task** is one contribution to the active Stage. It has an outcome that can \
be checked independently, or it produces evidence another Task consumes.

A **Step** is an action or working hypothesis for completing one Task. A Step \
can be revised when new evidence invalidates the tactic without changing the \
Task outcome.

**Evidence** is an observation produced by a tool, a check that ran, or state \
inspected in the current workspace. A remembered path, API or result is a \
hypothesis until current evidence confirms it.

**Complete** means every acceptance condition at that level has named evidence \
and no observed contradiction remains unresolved. **Blocked** means continuing \
requires a user-owned decision, missing authority or external state that an \
in-scope action cannot produce.

Keep three sources distinct. ``USER_LITERAL`` records what the user said or \
confirmed. ``DERIVED`` records planning choices and proposed checks. \
``OBSERVED_EVIDENCE`` records facts established during the run. Derived material \
can guide execution but cannot expand the Goal or acquire user authority.
"""


__all__ = ["PLANNING_VOCABULARY"]
