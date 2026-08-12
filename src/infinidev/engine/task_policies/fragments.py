"""Role- and phase-specific prompt fragments for selected task policies."""

from __future__ import annotations

from infinidev.engine.prompt_composition import ConditionalPromptFragment


def _fragment(
    id: str,
    policy_id: str,
    role: str,
    phase: str,
    content: str,
    *,
    operation: str = "",
    constraint: str = "",
    authority: str = "",
    model_routes: frozenset[str] = frozenset(),
    excluded_model_routes: frozenset[str] = frozenset(),
    priority: int = 50,
    version: int = 1,
) -> ConditionalPromptFragment:
    return ConditionalPromptFragment(
        id=id,
        policy_id=policy_id,
        content=content,
        roles=frozenset({role}),
        phases=frozenset({phase}),
        priority=priority,
        requires_operations=frozenset({operation}) if operation else frozenset(),
        requires_constraints=frozenset({constraint}) if constraint else frozenset(),
        requires_authority=frozenset({authority}) if authority else frozenset(),
        excludes_constraints=(
            frozenset({"read_only"}) if authority == "modify" else frozenset()
        ),
        model_routes=model_routes,
        excluded_model_routes=excluded_model_routes,
        version=version,
    )


TASK_METHOD_FRAGMENTS: tuple[ConditionalPromptFragment, ...] = (
    _fragment(
        "compatibility.planner", "compatibility.preserve_public_api", "planner", "plan",
        "Treat the public API as an invariant. Name affected public callers and put an explicit "
        "compatibility check in the Step that could cross that boundary.",
        constraint="preserve_public_api", priority=90,
    ),
    _fragment(
        "compatibility.developer", "compatibility.preserve_public_api", "developer", "execute",
        "Preserve the public API. Inspect public callers before changing a signature or observable "
        "contract, and run the narrowest compatibility check that exercises that boundary.",
        constraint="preserve_public_api", priority=90,
    ),
    _fragment(
        "compatibility.reviewer", "compatibility.preserve_public_api", "reviewer", "review",
        "Compare the submitted behavior and signatures with the established public contract. "
        "Reject only a demonstrated incompatibility, citing the affected caller or check.",
        constraint="preserve_public_api", priority=90,
    ),
    _fragment(
        "review.planner", "review.read_only", "planner", "plan",
        "Plan a read-only audit. Organize Steps around evidence collection and a "
        "prioritized report; do not create implementation or fix Steps.",
        operation="review", priority=80,
    ),
    _fragment(
        "review.developer", "review.read_only", "developer", "execute",
        "Work in read-only review mode. Inspect the relevant source and report "
        "substantiated findings by impact; do not implement corrections or turn "
        "speculation into a blocker.",
        operation="review", priority=80,
    ),
    _fragment(
        "review.reviewer", "review.read_only", "reviewer", "review",
        "Evaluate the audit report for evidence quality, severity calibration, and "
        "scope discipline. Do not demand implementation work from a read-only task.",
        operation="review", priority=80,
    ),
    _fragment(
        "bugfix.planner", "bugfix.root_cause", "planner", "plan",
        "Plan from failure evidence to root cause, then to the smallest sufficient "
        "repair. Include a focused regression check that distinguishes the fix from "
        "merely hiding the symptom.",
        operation="bugfix", authority="modify", priority=70,
    ),
    _fragment(
        "bugfix.developer", "bugfix.root_cause", "developer", "execute",
        "Repair the narrowest demonstrated contract violation. Keep unrelated behavior "
        "unchanged and validate the reproduced failure plus any directly affected contract.",
        operation="bugfix", authority="modify", priority=70, version=3,
    ),
    _fragment(
        "bugfix.reviewer", "bugfix.root_cause", "reviewer", "review",
        "Verify that the submitted change addresses the demonstrated cause rather than only the "
        "symptom, and that the regression check would fail without the correction.",
        operation="bugfix", authority="modify", priority=70,
    ),
    _fragment(
        "refactor.planner", "refactor.preserve_behavior", "planner", "plan",
        "Plan from an observable baseline. Identify callers and tests, split structural "
        "changes into reversible boundaries, and attach a behavior-preservation check "
        "to each boundary.",
        operation="refactor", authority="modify", priority=60,
    ),
    _fragment(
        "refactor.developer", "refactor.preserve_behavior", "developer", "execute",
        "Establish the current observable behavior, then make incremental structural "
        "changes without altering outputs or contracts. Re-run the narrowest relevant "
        "check after each boundary moves.",
        operation="refactor", authority="modify", priority=60,
    ),
    _fragment(
        "refactor.reviewer", "refactor.preserve_behavior", "reviewer", "review",
        "Review behavioral equivalence first: compare public contracts, callers, and test evidence "
        "before judging the new structure. Treat an observable regression as blocking.",
        operation="refactor", authority="modify", priority=60,
    ),
    _fragment(
        "feature.planner", "feature.contract_first", "planner", "plan",
        "Define the new user-visible contract and acceptance path before implementation. Identify "
        "integration and compatibility boundaries and plan the smallest end-to-end slice.",
        operation="feature", authority="modify", priority=50,
    ),
    _fragment(
        "feature.developer", "feature.contract_first", "developer", "execute",
        "Implement the smallest complete slice of the requested new capability. Follow existing "
        "integration patterns, avoid speculative architecture, and exercise the new "
        "path end to end.",
        operation="feature", authority="modify", priority=50,
    ),
    _fragment(
        "feature.reviewer", "feature.contract_first", "reviewer", "review",
        "Map the new user workflow and each explicit acceptance criterion to submitted evidence. "
        "Check integration and failure boundaries without inventing additional product "
        "requirements.",
        operation="feature", authority="modify", priority=50,
    ),
    _fragment(
        "performance.planner", "performance.measure_first", "planner", "plan",
        "Plan a representative measurement before drawing a performance conclusion. If the task "
        "authorizes optimization, make the change depend on an observed bottleneck and compare "
        "the same workload afterwards; otherwise plan evidence collection and a report only.",
        operation="performance", priority=55, version=2,
    ),
    _fragment(
        "performance.developer", "performance.measure_first", "developer", "execute",
        "Measure or reproduce a representative baseline and identify the observed bottleneck. "
        "When modification is authorized, optimize that bottleneck, preserve correctness, and "
        "compare the identical workload; in read-only work, report the measurements without "
        "tuning the implementation.",
        operation="performance", priority=55, version=2,
    ),
    _fragment(
        "performance.reviewer", "performance.measure_first", "reviewer", "review",
        "Require comparable before/after measurements and unchanged correctness. Do not accept a "
        "performance claim based only on code shape or an unrepresentative workload.",
        operation="performance", priority=55, version=2,
    ),
    _fragment(
        "research.planner", "research.evidence_first", "planner", "plan",
        "Plan around answerable research questions, primary sources, and explicit "
        "decision criteria. Separate facts, inferences, and unresolved questions; do "
        "not create coding Steps unless the literal task also authorizes implementation.",
        operation="research", priority=45,
    ),
    _fragment(
        "research.researcher", "research.evidence_first", "researcher", "investigate",
        "Gather primary, current evidence for each consequential claim. Cite sources near claims, "
        "distinguish inference from fact, compare credible alternatives, and expose "
        "remaining gaps.",
        operation="research", priority=45,
    ),
    _fragment(
        "research.developer", "research.evidence_first", "developer", "execute",
        "Treat this as evidence-first work. Gather and cite reliable material, "
        "distinguish facts from inferences, and do not modify files unless the literal "
        "request independently grants that authority.",
        operation="research", priority=45,
    ),
    _fragment(
        "research.reviewer", "research.evidence_first", "reviewer", "review",
        "Review source quality, recency, claim-to-citation support, competing evidence, "
        "and the separation of verified facts from inference. Do not require code from "
        "a research-only task.",
        operation="research", priority=45,
    ),
)


__all__ = ["TASK_METHOD_FRAGMENTS"]
