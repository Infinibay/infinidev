"""Canonical, bounded task-policy registry."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class TaskPolicy:
    """A versioned prompt fragment selected from structured task signals."""

    id: str
    version: int
    operations: frozenset[str]
    constraints: frozenset[str]
    roles: frozenset[str]
    phases: frozenset[str]
    priority: int
    max_utf8_bytes: int
    content: str
    incompatible_with: frozenset[str] = frozenset()
    requires_modify: bool = False
    forbids_modify: bool = False

    @property
    def content_hash(self) -> str:
        """Stable identity for replay and offline calibration artifacts."""
        payload = f"{self.id}\0{self.version}\0{self.content}".encode()
        return hashlib.sha256(payload).hexdigest()


POLICIES: tuple[TaskPolicy, ...] = (
    TaskPolicy(
        id="compatibility.preserve_public_api", version=1,
        operations=frozenset(), constraints=frozenset({"preserve_public_api"}),
        roles=frozenset({"planner", "developer", "reviewer"}),
        phases=frozenset({"investigate", "plan", "execute", "review"}), priority=90,
        max_utf8_bytes=900,
        content=("Treat the public API as an invariant. Identify public callers and contracts "
                 "before editing, avoid incompatible signature or behavior changes, and verify "
                 "compatibility explicitly."),
    ),
    TaskPolicy(
        id="review.read_only", version=1,
        operations=frozenset({"review"}), constraints=frozenset({"read_only"}),
        roles=frozenset({"planner", "developer", "reviewer"}),
        phases=frozenset({"investigate", "plan", "execute", "review"}), priority=80,
        max_utf8_bytes=1000,
        content=("Perform a read-only review. Find concrete defects, regressions, and risks; "
                 "prioritize findings by impact and cite evidence. Separate verified defects "
                 "from assumptions about unseen callers or context; label open questions and "
                 "never elevate speculation to a blocker. Do not implement fixes. State "
                 "explicitly when no material defect is found."),
        forbids_modify=True,
    ),
    TaskPolicy(
        id="bugfix.root_cause", version=1,
        operations=frozenset({"bugfix"}), constraints=frozenset(),
        roles=frozenset({"planner", "developer", "reviewer"}),
        phases=frozenset({"investigate", "plan", "execute", "review"}), priority=70,
        max_utf8_bytes=1000,
        content=("Reproduce or establish evidence for the failure, distinguish the symptom "
                 "from its root cause, make the smallest sufficient correction, and run or add "
                 "a focused regression check."),
        requires_modify=True,
    ),
    TaskPolicy(
        id="refactor.preserve_behavior", version=1,
        operations=frozenset({"refactor"}), constraints=frozenset(),
        roles=frozenset({"planner", "developer", "reviewer"}),
        phases=frozenset({"investigate", "plan", "execute", "review"}), priority=60,
        max_utf8_bytes=1100,
        content=("Preserve observable behavior. Establish a baseline, identify callers and "
                 "tests, make incremental structural changes, and rerun the narrowest relevant "
                 "verification whenever a boundary changes."),
        requires_modify=True,
    ),
    TaskPolicy(
        id="feature.contract_first", version=1,
        operations=frozenset({"feature"}), constraints=frozenset(),
        roles=frozenset({"planner", "developer", "reviewer"}),
        phases=frozenset({"plan", "execute", "review"}), priority=50,
        max_utf8_bytes=1100,
        content=(
            "Define the requested behavior and acceptance criteria before implementation. "
            "Inspect integration and compatibility boundaries, avoid unnecessary "
            "architecture, and verify the new path end to end."
        ),
        requires_modify=True,
    ),
    TaskPolicy(
        id="research.evidence_first", version=1,
        operations=frozenset({"research"}), constraints=frozenset(),
        roles=frozenset({"planner", "developer", "reviewer"}),
        phases=frozenset({"investigate", "plan", "execute", "review"}), priority=40,
        max_utf8_bytes=1050,
        content=("Separate verified facts, inferences, and open questions. Prefer primary "
                 "sources, cite evidence near each claim, and do not modify files unless the "
                 "literal request grants modification authority."),
    ),
    TaskPolicy(
        id="performance.measure_first", version=2,
        operations=frozenset({"performance"}), constraints=frozenset(),
        roles=frozenset({"planner", "developer", "reviewer"}),
        phases=frozenset({"investigate", "plan", "execute", "review"}), priority=45,
        max_utf8_bytes=1000,
        content=("Measure or reproduce a representative baseline and identify the actual "
                 "bottleneck. If changes are authorized, preserve correctness and compare the "
                 "same workload afterwards; otherwise report the evidence without changing the "
                 "implementation."),
    ),
)

POLICY_BY_ID = {policy.id: policy for policy in POLICIES}
