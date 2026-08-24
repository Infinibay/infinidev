"""GraphEngineAdapter — runs an escalated task as a work graph.

Ties the pieces together: seed a graph from the escalation, then loop
(select node → build capsule → execute leaf → feed evidence back → repeat)
until the completion gate closes the run (docs/GRAPH_ENGINE_BETA_DESIGN.md
§8.3, §13 phases 5–7).

The live leaf executor runs the existing LoopEngine as a bounded micro-episode.
Its result returns as evidence for the active node (§8.5): the run never
leaves Graph or enters the Stage Planner. Tests may still inject a small
executor to exercise graph semantics in isolation.

This adapter never reads or writes the pipeline's staged state.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

from infinidev.engine.engines.base import (
    EngineResult,
    STATUS_BLOCKED,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    TransitionRequest,
    get_loop_status,
    normalize_loop_status,
)
from infinidev.engine.engines.graph import completion
from infinidev.engine.engines.graph.context import build_capsule, render_capsule
from infinidev.engine.engines.graph.domain import (
    EDGE_DECOMPOSES_INTO,
    EDGE_REQUIRES,
    EDGE_SATISFIES,
    GraphState,
    Lifecycle,
    new_node_id,
)
from infinidev.engine.engines.graph.ops import (
    ActivateNodeOp,
    AttachEvidenceOp,
    EdgeSpec,
    GraphPatchOp,
    GraphOp,
    NodeSpec,
    ResolveNodeOp,
    ReviseGoalOp,
    SuspendNodeOp,
)
from infinidev.engine.engines.graph.reducer import GraphInvariantError, reduce
from infinidev.engine.engines.graph.scheduler import SchedulerLimits, select_next

logger = logging.getLogger(__name__)

#: A leaf executor receives the rendered capsule text plus a budget dict and
#: returns the leaf's result text. Injection is for tests and specialised
#: embedders; live coordinator runs use the LoopEngine path below.
LeafExecutor = Callable[[str, dict[str, Any]], str]

#: Safety fuse for the number of leaf executions in one run. Distinct from the
#: design's token/tool budgets; it only guarantees the loop terminates.
_DEFAULT_MAX_LEAF_RUNS = 12

# Internal, non-terminal leaf outcome. EngineResult intentionally has no
# ``interrupted`` status because a Graph budget boundary is scheduler state,
# not a user-visible task result: checkpoint the node and let the existing
# revisit/leaf-run fuses decide whether another micro-episode may resume it.
_LEAF_INTERRUPTED = "interrupted"


class GraphEngineAdapter:
    """Execute an escalated task as a work graph."""

    name = "graph_beta"

    def __init__(
        self,
        executor: LeafExecutor | None = None,
        persistence: Any | None = None,
        limits: SchedulerLimits | None = None,
        max_leaf_runs: int = _DEFAULT_MAX_LEAF_RUNS,
    ) -> None:
        self._executor = executor
        self._persistence = persistence
        self._limits = limits or SchedulerLimits()
        self._max_leaf_runs = max_leaf_runs

    # ── graph mutation (via persistence when present, else in-memory) ──

    def _apply(self, state: GraphState, op: GraphOp):
        if self._persistence is not None:
            return self._persistence.apply(state, op)
        new_state, _events = reduce(state, op)
        return new_state, []

    # ── seeding ────────────────────────────────────────────────────────

    def _seed_state(
        self, run_id: str, session_id: str, escalation: Any
    ) -> GraphState:
        """Build the initial graph from the grounded scope when available."""
        request = getattr(escalation, "user_request", "") or ""

        state = GraphState(run_id=run_id, session_id=session_id)
        state, _ = self._apply(
            state,
            ReviseGoalOp(text=request, classification="new_requirement"),
        )

        scope_items = self._scope_items(escalation, request)
        add_nodes: list[NodeSpec] = []
        add_edges: list[EdgeSpec] = []
        work_ids: list[str] = []
        verification_ids: list[str] = []
        for index, item in enumerate(scope_items):
            requirement_id = new_node_id("req")
            is_verification = self._is_verification_scope(item)
            is_evidence_only = self._is_evidence_scope(item)
            executable_id = new_node_id("verify" if is_verification else "work")
            if is_verification:
                verification_ids.append(executable_id)
            else:
                work_ids.append(executable_id)
            priority = float(len(scope_items) - index)
            add_nodes.extend([
                NodeSpec(
                    node_id=requirement_id,
                    node_type="requirement",
                    title=item[:120],
                    objective=item,
                    expected_outcome=item,
                    priority=priority,
                ),
                NodeSpec(
                    node_id=executable_id,
                    node_type="verification" if is_verification else "work",
                    title=(
                        f"Verify: {item}"
                        if is_verification
                        else f"Investigate: {item}"
                        if is_evidence_only
                        else f"Implement: {item}"
                    )[:120],
                    objective=item,
                    expected_outcome=item,
                    priority=priority,
                    payload={
                        "evidence_only": is_evidence_only,
                        "deferred_scope": [
                            other for other in scope_items if other != item
                        ],
                    },
                ),
            ])
            add_edges.extend([
                EdgeSpec(
                    source=requirement_id,
                    target=executable_id,
                    edge_type=EDGE_DECOMPOSES_INTO,
                ),
                EdgeSpec(
                    source=executable_id,
                    target=requirement_id,
                    edge_type=EDGE_SATISFIES,
                ),
            ])

        if verification_ids:
            add_edges.extend(
                EdgeSpec(
                    source=verification_id,
                    target=work_id,
                    edge_type=EDGE_REQUIRES,
                )
                for verification_id in verification_ids
                for work_id in work_ids
            )
        elif len(work_ids) > 1:
            verification_id = new_node_id("verify")
            add_nodes.append(NodeSpec(
                node_id=verification_id,
                node_type="verification",
                title="Verify the integrated goal",
                objective=(
                    "Verify the combined implementation against the complete "
                    "user request. Run relevant tests and correct integration "
                    "regressions before resolving this node."
                ),
                expected_outcome=(
                    "All completed branches integrate and the complete user "
                    "request is verified."
                ),
            ))
            add_edges.extend(
                EdgeSpec(
                    source=verification_id,
                    target=work_id,
                    edge_type=EDGE_REQUIRES,
                )
                for work_id in work_ids
            )

        patch = GraphPatchOp(
            add_nodes=add_nodes,
            add_edges=add_edges,
            rationale=(
                "Seed graph from grounded in-scope work."
                if len(scope_items) > 1
                else "Seed graph from the escalated request."
            ),
            based_on_revision=state.revision,
        )
        state, _ = self._apply(state, patch)
        return state

    @staticmethod
    def _is_verification_scope(item: str) -> bool:
        """Whether an elaborated scope item is an explicit validation action."""
        lowered = item.lstrip().lower()
        return lowered.startswith((
            "run ", "verify ", "validate ", "test ", "execute ",
        ))

    @staticmethod
    def _is_evidence_scope(item: str) -> bool:
        """Whether a scope item asks for grounding rather than mutation."""
        lowered = item.lstrip().lower()
        return lowered.startswith((
            "inspect ", "analyze ", "analyse ", "audit ", "explore ",
            "read ", "identify ", "determine ", "understand ",
            "review the current ", "review current ",
        ))

    def _scope_items(self, escalation: Any, request: str) -> list[str]:
        """Return bounded work branches from an already elaborated spec.

        The elaborator is useful evidence, not a graph schema authority. It
        sometimes combines several independent changes into one semicolon-
        separated paragraph or emits orchestration instructions as if they
        were repository deliverables. Normalize those shapes before seeding.
        """
        spec = getattr(escalation, "grounded_spec", None)
        raw_items = list(getattr(spec, "in_scope", None) or [])
        items = self._literal_scope_items(request)
        if not items:
            for raw in raw_items:
                item = " ".join(str(raw).split()).strip(" -")
                for part in self._split_compound_scope(item):
                    if (
                        part
                        and part not in items
                        and not self._is_graph_meta_scope(part)
                    ):
                        items.append(part)
        if not items:
            return [request]

        # A derived scope cannot introduce repository operations that the
        # literal request never authorized. In particular, models sometimes
        # reinterpret "independent Graph branches" as real Git branches.
        # Keep Git work only when the user actually named Git.
        if "git" not in request.lower():
            items = [item for item in items if "git branch" not in item.lower()]

        # Elaborators commonly emit a generic "inspect/read existing code"
        # preamble before concrete implementation scopes. Turning that into a
        # standalone LLM leaf repeats discovery, consumes an entire branch
        # budget, and contributes no independently reviewable outcome. For an
        # implementation request with at least two real downstream actions,
        # keep discovery inside those actions instead of scheduling it as a
        # sibling deliverable. Analysis-first user requests still retain their
        # evidence nodes.
        request_verb = (
            request.lstrip().split(maxsplit=1)[0].lower()
            if request.strip()
            else ""
        )
        implementation_verbs = {
            "add", "build", "change", "correct", "create", "fix",
            "implement", "refactor", "replace", "update",
        }
        actionable = [item for item in items if not self._is_evidence_scope(item)]
        if request_verb in implementation_verbs and len(actionable) >= 2:
            items = actionable

        if len(items) == 1:
            return [request]

        limit = max(2, self._limits.max_open_branches)
        # Reserve one branch slot for the synthetic integration verifier when
        # the literal did not provide an explicit verification item.
        item_limit = (
            limit
            if any(self._is_verification_scope(item) for item in items)
            else max(1, limit - 1)
        )
        if len(items) > item_limit:
            head = items[: item_limit - 1]
            tail = "; ".join(items[item_limit - 1:])
            items = [*head, f"Complete remaining in-scope work: {tail}"]
        return items

    @staticmethod
    def _split_compound_scope(item: str) -> list[str]:
        """Split independent semicolon clauses without fragmenting prose."""
        if item.count(";") < 2:
            return [item] if item else []
        parts = [part.strip() for part in re.split(r";\s*", item) if part.strip()]
        if len(parts) < 3 or any(len(part) < 20 for part in parts):
            return [item]
        return [
            re.sub(r"^(?:and|then)\s+", "", part, flags=re.IGNORECASE)
            for part in parts
        ]

    @staticmethod
    def _literal_scope_items(request: str) -> list[str]:
        """Recover explicitly enumerated requirements from the user literal.

        The model-derived spec can collapse ``First / Second / Third`` (or a
        numbered list) into one broad implementation sentence. Literal
        enumeration is stronger topology evidence, so use it directly and
        stop the final item before quality constraints or the integration
        gate.
        """
        ordinal = re.compile(
            r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth)\s*[, :]\s*",
            re.IGNORECASE,
        )
        numbered = re.compile(r"(?<!\w)(?:\(\d+\)|\d+[.)])\s+")
        matches = list(ordinal.finditer(request))
        if len(matches) < 2:
            matches = list(numbered.finditer(request))
        if len(matches) < 2:
            return []

        stop = re.compile(
            r"\b(?:use idiomatic|do not |don't |final gate|after all |"
            r"verify with|verification command|quality gate)\b",
            re.IGNORECASE,
        )
        items: list[str] = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(request)
            item = request[match.end():end].strip(" .;:-")
            if index + 1 == len(matches):
                item = stop.split(item, maxsplit=1)[0].strip(" .;:-")
            if len(item) >= 20:
                items.append(item)
        return items if len(items) >= 2 else []

    @staticmethod
    def _literal_constraints(request: str) -> list[str]:
        """Keep explicit negative constraints on every isolated work leaf."""
        constraints: list[str] = []
        for match in re.finditer(
            r"\b(?:do not|don't)\s+[^.\n]+[.]?",
            request,
            flags=re.IGNORECASE,
        ):
            value = " ".join(match.group(0).split()).rstrip(".")
            if value and value not in constraints:
                constraints.append(value)
        return constraints

    @staticmethod
    def _is_graph_meta_scope(item: str) -> bool:
        """Whether an item configures Graph instead of changing the workspace."""
        lowered = item.lower()
        return (
            "graph work node" in lowered
            and any(word in lowered for word in ("represent", "separate", "own"))
        )

    # ── helpers ────────────────────────────────────────────────────────

    def _requirements_satisfied_by(self, state: GraphState, node_id: str) -> list[str]:
        requirement_ids = []
        for edge in state.edges_from(node_id):
            if edge.edge_type != EDGE_SATISFIES:
                continue
            target = state.nodes.get(edge.target)
            if target is not None and target.node_type == "requirement":
                requirement_ids.append(target.node_id)
        return requirement_ids

    def _run_live_leaf(
        self,
        *,
        capsule_text: str,
        budget: dict[str, Any],
        node: Any,
        kwargs: dict[str, Any],
        preserve_file_tracker: bool,
        resume_leaf: bool = False,
    ) -> tuple[str, str]:
        """Execute one Graph node through the supplied LoopEngine."""
        from infinidev.config.settings import settings
        from infinidev.engine.analysis.plan import Plan
        from infinidev.engine.engines.task import _bootstrap_step
        from infinidev.engine.orchestration import pipeline as pipeline_mod
        from infinidev.engine.orchestration.staged_pipeline import (
            _goal_from_escalation,
        )
        from infinidev.engine.orchestration.task_renderer import render_task_xml
        from infinidev.engine.orchestration.task_schema import task_from_free_text
        from infinidev.prompts.flows import get_flow_config
        from infinidev.prompts.profiles import EffectivePromptConfiguration

        prompt_configuration = (
            kwargs.get("prompt_configuration")
            or EffectivePromptConfiguration.compile()
        )
        escalation = kwargs["escalation"]
        agent = kwargs["agent"]
        engine = kwargs["engine"]
        hooks = kwargs["hooks"]
        session_id = kwargs["session_id"]
        goal = _goal_from_escalation(escalation)

        node_objective = (getattr(node, "objective", "") or "").strip()
        is_goal_verification = getattr(node, "node_type", "") == "verification"
        is_evidence_only = bool(
            getattr(node, "payload", {}).get("evidence_only")
        )
        if is_goal_verification:
            literal_description = (
                f"Verify the complete user goal: {goal.user_request}\n\n"
                f"Active Graph node: {node_objective}"
            )
        else:
            literal_description = f"Active Graph node: {node_objective}"
        if len(literal_description.strip()) < 20:
            literal_description = f"User request (verbatim): {literal_description}"
        title = (getattr(node, "title", "") or "Graph node").strip()[:120]
        if len(title) < 5:
            title = f"{title} task"[:120]
        deferred_scope = list(getattr(node, "payload", {}).get("deferred_scope", []))
        spec_out_of_scope = list(
            getattr(getattr(escalation, "grounded_spec", None), "out_of_scope", None)
            or []
        )
        leaf_out_of_scope = spec_out_of_scope + [
            f"Sibling Graph branch; do not implement in this leaf: {item}"
            for item in deferred_scope
        ]
        literal_constraints = self._literal_constraints(goal.user_request)
        leaf_acceptance = (
            list(goal.acceptance_criteria) or None
            if is_goal_verification
            else None
        )
        leaf_derived = (
            list(goal.derived_verification_criteria)
            if is_goal_verification
            else [
                criterion for criterion in [
                    (getattr(node, "expected_outcome", "") or "").strip()
                ]
                if criterion
            ]
        )
        structured_task = task_from_free_text(
            literal_description,
            title=title,
            kind=(
                "verification"
                if is_goal_verification
                else "investigation"
                if is_evidence_only
                else "feature"
            ),
            acceptance_criteria=leaf_acceptance,
            derived_verification_criteria=leaf_derived,
            out_of_scope=leaf_out_of_scope,
            constraints=[
                *literal_constraints,
                *(
                    [
                        "Work only on the active Graph node; do not execute "
                        "sibling branches."
                    ]
                    if deferred_scope
                    else []
                ),
            ],
            task_profile=escalation.task_profile,
        )
        leaf_plan = Plan(
            overview=(
                "The Graph scheduler owns global scope and dependencies. "
                "This rolling plan covers only the active Graph node."
            ),
            steps=[_bootstrap_step(structured_task)],
            rolling_horizon_limit=3,
        )
        approach = (
            '<approach authority="DERIVED">\n'
            "Execute only the active Graph node. Work incrementally, verify the "
            "outcome, and finish with step_complete when this node is satisfied. "
            "Leave every sibling branch to the Graph scheduler. Do not invoke "
            "Stage planning or expand the active node; the Graph scheduler "
            "retains orchestration authority.\n"
            "</approach>"
        )
        # A work leaf must not receive the complete parent goal as its active
        # review contract. Doing so makes valid partial work fail because a
        # later sibling is still pending, and the review-rework loop then edits
        # that sibling from the wrong branch. The integration verifier is the
        # only leaf that owns the complete goal and its deterministic gate.
        prompt_context = (
            capsule_text
            if is_goal_verification
            else render_task_xml(structured_task)
        )
        task_prompt = (
            f"{prompt_context}\n\n{approach}",
            get_flow_config("develop").expected_output,
        )
        task_prompt = pipeline_mod._run_gather_phase(
            user_input=getattr(node, "title", "Graph node"),
            agent=agent,
            task_prompt=task_prompt,
            session_id=session_id,
            force_gather=kwargs.get("force_gather", False),
            hooks=hooks,
            prompt_configuration=prompt_configuration,
        )

        configured_tool_calls = int(
            budget.get("max_tool_calls", settings.REACT_MAX_TOOL_CALLS)
        )
        positive_tool_limits = [
            limit
            for limit in (settings.REACT_MAX_TOOL_CALLS, configured_tool_calls)
            if limit > 0
        ]
        max_tool_calls = min(positive_tool_limits) if positive_tool_limits else 0
        max_prompt_tokens = int(
            budget.get("token_budget", settings.GRAPH_NODE_TOKEN_BUDGET)
        )
        hooks.on_phase("execute")
        hooks.on_status(
            "info",
            f"Graph node: {title} (budget {max_tool_calls} tool calls)",
        )
        agent.activate_context(session_id=session_id)
        try:
            resume_state = self._resume_leaf_state(engine) if resume_leaf else None
            result = engine.execute(
                agent=agent,
                task_prompt=task_prompt,
                verbose=True,
                initial_plan=leaf_plan,
                initial_attachments=(
                    list(escalation.attachments) if escalation.attachments else None
                ),
                task=structured_task,
                max_iterations=settings.REACT_MAX_ITERATIONS,
                max_total_tool_calls=max_tool_calls,
                max_prompt_tokens=max_prompt_tokens,
                preserve_file_tracker=preserve_file_tracker,
                resume_state=resume_state,
                skip_plan=False,
                allow_plan_mutation=False,
                prompt_configuration=prompt_configuration,
            )
        finally:
            agent.deactivate()

        result = (
            result.strip()
            if result and result.strip()
            else "Done. (no additional output)"
        )
        if getattr(engine, "is_cancelled", False):
            return result, STATUS_CANCELLED

        loop_status = get_loop_status(engine)
        status = normalize_loop_status(loop_status)
        if loop_status == "exhausted":
            return result, _LEAF_INTERRUPTED
        if status == STATUS_FAILED and loop_status != "failed":
            hooks.on_status(
                "error",
                "Graph leaf returned an empty or unknown terminal status; "
                "failing closed instead of resolving the node.",
            )
        if status != STATUS_COMPLETED:
            return result, status

        if getattr(node, "payload", {}).get("evidence_only"):
            if engine.has_file_changes():
                return (
                    "Evidence-only Graph node changed files; refusing to "
                    "resolve it outside its declared scope.",
                    STATUS_BLOCKED,
                )
            return result, STATUS_COMPLETED

        result = pipeline_mod._run_review_phase(
            engine=engine,
            agent=agent,
            session_id=session_id,
            task_prompt=task_prompt,
            result=result,
            reviewer=kwargs.get("reviewer"),
            hooks=hooks,
            acceptance_criteria=leaf_acceptance,
            derived_verification_criteria=leaf_derived,
            task=structured_task,
            max_iterations=settings.REACT_MAX_ITERATIONS,
            max_total_tool_calls=max_tool_calls,
            rework_execute_kwargs={
                "skip_plan": False,
                "allow_plan_mutation": False,
                "max_prompt_tokens": max_prompt_tokens,
            },
            run_verification=is_goal_verification,
            prompt_configuration=prompt_configuration,
        )
        review_status = get_loop_status(engine)
        review_outcome = normalize_loop_status(review_status)
        if getattr(engine, "is_cancelled", False):
            return result, STATUS_CANCELLED
        if review_status == "exhausted":
            return result, _LEAF_INTERRUPTED
        if review_outcome == STATUS_FAILED and review_status != "failed":
            hooks.on_status(
                "error",
                "Graph leaf review returned an empty or unknown terminal status; "
                "failing closed instead of resolving the node.",
            )
        return result, review_outcome

    @staticmethod
    def _resume_leaf_state(engine: Any) -> dict[str, Any] | None:
        """Carry leaf evidence into a fresh bounded micro-episode.

        The plan, edits, notes, and compact history form the checkpoint.
        Resource counters are episode-local fuses, so carrying them forward
        would make the resumed loop exit before its first model turn. Graph
        calls this only for an immediate retry of the same node; sibling
        leaves always start from isolated state.
        """
        previous = getattr(engine, "_last_state", None)
        if previous is None:
            return None
        state = previous.model_dump(mode="json")
        for field in (
            "iteration_count",
            "total_tool_calls",
            "total_tokens",
            "total_prompt_tokens",
            "total_completion_tokens",
            "last_prompt_tokens",
            "last_completion_tokens",
            "cache_creation_tokens",
            "cache_read_tokens",
            "cached_tokens",
            "tool_calls_since_last_note",
        ):
            state[field] = 0
        state["prompt_composition_history"] = []
        state["request_payload_history"] = []
        return state

    # ── main loop ──────────────────────────────────────────────────────

    @staticmethod
    def _accumulate_leaf_observability(
        engine: Any,
        previous_tracker: Any | None,
        total_tool_calls: int,
        *,
        tracker_was_preserved: bool,
    ) -> tuple[Any | None, int]:
        """Aggregate branch changes and counters after branch-local review.

        Sibling work leaves stay isolated while they execute and are reviewed.
        Once a leaf closes, its tracker absorbs the older aggregate so the
        integration verifier and final task summary can observe every branch.
        Resumed leaves already merged the prior tracker in LoopEngine.
        """
        if engine is None:
            return previous_tracker, total_tool_calls

        current_tracker = getattr(engine, "_last_file_tracker", None)
        if current_tracker is None:
            if previous_tracker is not None:
                engine._last_file_tracker = previous_tracker
            current_tracker = previous_tracker
        elif (
            previous_tracker is not None
            and current_tracker is not previous_tracker
            and not tracker_was_preserved
        ):
            merge_from = getattr(current_tracker, "merge_from", None)
            if callable(merge_from):
                merge_from(previous_tracker)

        run_tool_calls = getattr(engine, "_last_total_tool_calls", 0)
        if isinstance(run_tool_calls, int):
            total_tool_calls += run_tool_calls
        engine._last_total_tool_calls = total_tool_calls
        return current_tracker, total_tool_calls

    def run(self, **kwargs: Any) -> EngineResult:
        from infinidev.config.settings import settings

        from infinidev.prompts.profiles import EffectivePromptConfiguration

        kwargs["prompt_configuration"] = (
            kwargs.get("prompt_configuration")
            or EffectivePromptConfiguration.compile()
        )
        escalation = kwargs.get("escalation")
        session_id = kwargs.get("session_id", "")
        run_id = kwargs.get("run_id") or f"graph_{id(self):x}"
        visits: dict[str, int] = {}
        leaf_runs = 0
        resume_node_id: str | None = None
        last_result = ""
        completed_results: list[tuple[str, str]] = []
        aggregate_tracker: Any | None = None
        total_tool_calls = 0

        def current_metrics() -> dict[str, Any]:
            return {
                "max_leaf_runs": self._max_leaf_runs,
                "leaf_runs": leaf_runs,
                "max_tool_calls": (
                    None
                    if settings.GRAPH_RUN_TOOL_BUDGET <= 0
                    else settings.GRAPH_RUN_TOOL_BUDGET
                ),
                "observed_tool_calls": total_tool_calls,
                "max_prompt_tokens_per_node": (
                    None
                    if settings.GRAPH_NODE_TOKEN_BUDGET <= 0
                    else settings.GRAPH_NODE_TOKEN_BUDGET
                ),
                "max_open_branches": self._limits.max_open_branches,
                "max_node_revisits": self._limits.max_node_revisits,
                "visited_nodes": len(visits),
                "node_visits": sum(visits.values()),
            }

        def remaining_run_tool_calls() -> int | None:
            configured = settings.GRAPH_RUN_TOOL_BUDGET
            if configured <= 0:
                return None
            return max(0, configured - total_tool_calls)

        try:
            state = self._seed_state(run_id, session_id, escalation)
        except GraphInvariantError as exc:
            return EngineResult(
                engine_name=self.name,
                status=STATUS_BLOCKED,
                user_message=f"Could not seed the work graph: {exc}",
                summary="seed failed",
                engine=kwargs.get("engine"),
                metrics=current_metrics(),
            )

        while True:
            assessment = completion.evaluate_goal(state)
            if assessment.status == "complete":
                return EngineResult(
                    engine_name=self.name,
                    status=STATUS_COMPLETED,
                    user_message=self._render_completed_results(
                        completed_results, last_result,
                    ),
                    summary="; ".join(assessment.reasons),
                    engine=kwargs.get("engine"),
                    state=state,
                    resume_token=session_id,
                    metrics=current_metrics(),
                )
            if assessment.status == "blocked":
                return EngineResult(
                    engine_name=self.name,
                    status=STATUS_BLOCKED,
                    user_message=(
                        "The graph is blocked: " + "; ".join(assessment.missing)
                    ),
                    summary="; ".join(assessment.reasons),
                    engine=kwargs.get("engine"),
                    state=state,
                    resume_token=session_id,
                    metrics=current_metrics(),
                )

            remaining_tool_calls = remaining_run_tool_calls()
            if remaining_tool_calls == 0:
                run_tool_budget = settings.GRAPH_RUN_TOOL_BUDGET
                return EngineResult(
                    engine_name=self.name,
                    status=STATUS_BLOCKED,
                    user_message=(
                        "The graph engine reached its run-wide tool-call fuse "
                        f"({total_tool_calls}/{run_tool_budget}) before completing."
                    ),
                    summary="run tool-call budget exhausted",
                    engine=kwargs.get("engine"),
                    state=state,
                    resume_token=session_id,
                    metrics=current_metrics(),
                    transition_request=TransitionRequest(
                        target="staged",
                        reason=(
                            "graph_run_tool_budget_exhausted: "
                            f"{total_tool_calls}/{run_tool_budget} tool calls"
                        ),
                    ),
                )

            if leaf_runs >= self._max_leaf_runs:
                return EngineResult(
                    engine_name=self.name,
                    status=STATUS_BLOCKED,
                    user_message=(
                        "The graph engine reached its leaf-execution fuse "
                        f"after {leaf_runs} runs without completing."
                    ),
                    summary="leaf-run budget exhausted",
                    engine=kwargs.get("engine"),
                    state=state,
                    resume_token=session_id,
                    metrics=current_metrics(),
                    transition_request=TransitionRequest(
                        target="staged",
                        reason="graph_leaf_budget_exhausted",
                    ),
                )

            resuming_leaf = resume_node_id is not None
            if resuming_leaf:
                resume_visits = visits.get(resume_node_id, 0)
                if resume_visits >= self._limits.max_node_revisits:
                    node = None
                    reason = (
                        f"checkpointed node {resume_node_id} exceeded its "
                        f"revisit budget ({self._limits.max_node_revisits})"
                    )
                else:
                    node = state.nodes.get(resume_node_id)
                    reason = f"resume checkpointed node {resume_node_id}"
                resume_node_id = None
            else:
                node, reason = select_next(
                    state, visits=visits, limits=self._limits
                )
            if node is None:
                return EngineResult(
                    engine_name=self.name,
                    status=STATUS_BLOCKED,
                    user_message=f"No runnable node left: {reason}",
                    summary=reason,
                    engine=kwargs.get("engine"),
                    state=state,
                    resume_token=session_id,
                    metrics=current_metrics(),
                )

            node_id = node.node_id
            try:
                state, _ = self._apply(
                    state, ActivateNodeOp(node_id=node_id, rationale=reason)
                )
                node_tool_limits = [
                    limit
                    for limit in (
                        settings.REACT_MAX_TOOL_CALLS,
                        remaining_tool_calls,
                    )
                    if limit is not None and limit > 0
                ]
                node_budget = {
                    "token_budget": settings.GRAPH_NODE_TOKEN_BUDGET,
                    "max_tool_calls": (
                        min(node_tool_limits) if node_tool_limits else 0
                    ),
                }
                capsule = build_capsule(
                    state,
                    node_id,
                    budget=node_budget,
                    selection_reason=reason,
                )
                capsule_text = render_capsule(capsule)
                if self._executor is None:
                    preserve_leaf_tracker = (
                        resuming_leaf or node.node_type == "verification"
                    )
                    result_text, leaf_status = self._run_live_leaf(
                        capsule_text=capsule_text,
                        budget=node_budget,
                        node=node,
                        kwargs=kwargs,
                        # Work siblings stay isolated. A resumed leaf needs its
                        # checkpoint, while the integration verifier needs the
                        # accumulated task diff from every completed branch.
                        preserve_file_tracker=preserve_leaf_tracker,
                        resume_leaf=resuming_leaf,
                    )
                else:
                    result_text = self._executor(capsule_text, node_budget)
                    leaf_status = STATUS_COMPLETED
            except GraphInvariantError as exc:
                return EngineResult(
                    engine_name=self.name,
                    status=STATUS_BLOCKED,
                    user_message=f"Graph mutation rejected: {exc}",
                    summary=str(exc),
                    engine=kwargs.get("engine"),
                    state=state,
                    resume_token=session_id,
                    metrics=current_metrics(),
                )

            if self._executor is None:
                aggregate_tracker, total_tool_calls = (
                    self._accumulate_leaf_observability(
                        kwargs.get("engine"),
                        aggregate_tracker,
                        total_tool_calls,
                        tracker_was_preserved=preserve_leaf_tracker,
                    )
                )

            if leaf_status == _LEAF_INTERRUPTED:
                try:
                    state, _ = self._apply(
                        state,
                        SuspendNodeOp(
                            node_id=node_id,
                            reason="leaf execution budget exhausted",
                            checkpoint=result_text,
                        ),
                    )
                except GraphInvariantError as exc:
                    return EngineResult(
                        engine_name=self.name,
                        status=STATUS_BLOCKED,
                        user_message=f"Could not checkpoint interrupted leaf: {exc}",
                        summary=str(exc),
                        engine=kwargs.get("engine"),
                        state=state,
                        resume_token=session_id,
                        metrics=current_metrics(),
                    )
                visits[node_id] = visits.get(node_id, 0) + 1
                leaf_runs += 1
                resume_node_id = node_id
                continue

            if leaf_status != STATUS_COMPLETED:
                return EngineResult(
                    engine_name=self.name,
                    status=leaf_status,
                    user_message=result_text,
                    summary=f"Graph leaf {node_id} closed {leaf_status}.",
                    engine=kwargs.get("engine"),
                    state=state,
                    resume_token=session_id,
                    metrics=current_metrics(),
                )

            last_result = result_text or last_result
            completed_results.append((node.title or node_id, result_text))

            # Feed the leaf outcome back as evidence and resolve the node.
            evidence_id = new_node_id("evidence")
            try:
                state, _ = self._apply(
                    state,
                    GraphPatchOp(
                        add_nodes=[
                            NodeSpec(
                                node_id=evidence_id,
                                node_type="evidence",
                                title=f"Evidence for {node_id}",
                                objective=result_text,
                            )
                        ],
                        rationale="Record the leaf's outcome as evidence.",
                        based_on_revision=state.revision,
                    ),
                )
                state, _ = self._apply(
                    state,
                    AttachEvidenceOp(
                        node_id=node_id, evidence_id=evidence_id,
                        summary=result_text,
                    ),
                )
                state, _ = self._apply(
                    state,
                    ResolveNodeOp(
                        node_id=node_id, evidence_ids=[evidence_id],
                        outcome=result_text, verdict="confirmed",
                    ),
                )
                for requirement_id in self._requirements_satisfied_by(state, node_id):
                    state, _ = self._apply(
                        state,
                        ResolveNodeOp(
                            node_id=requirement_id, evidence_ids=[evidence_id],
                            outcome=result_text, verdict="confirmed",
                        ),
                    )
            except GraphInvariantError as exc:
                return EngineResult(
                    engine_name=self.name,
                    status=STATUS_BLOCKED,
                    user_message=f"Could not record evidence: {exc}",
                    summary=str(exc),
                    engine=kwargs.get("engine"),
                    state=state,
                    resume_token=session_id,
                    metrics=current_metrics(),
                )

            visits[node_id] = visits.get(node_id, 0) + 1
            leaf_runs += 1

    @staticmethod
    def _render_completed_results(
        completed_results: list[tuple[str, str]], last_result: str,
    ) -> str:
        """Return all branch outcomes instead of hiding every result but the last."""
        if not completed_results:
            return last_result or "Goal completed."
        if len(completed_results) == 1:
            return completed_results[0][1] or "Goal completed."
        sections = [
            f"### {title}\n\n{result or 'Completed.'}"
            for title, result in completed_results
        ]
        return "Graph completed all work nodes:\n\n" + "\n\n".join(sections)


__all__ = ["GraphEngineAdapter", "LeafExecutor"]
