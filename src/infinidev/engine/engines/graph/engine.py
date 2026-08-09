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
            if edge.edge_type is not EDGE_SATISFIES:
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
    ) -> tuple[str, str]:
        """Execute one Graph node through the supplied LoopEngine."""
        from infinidev.config.settings import settings
        from infinidev.engine.orchestration import pipeline as pipeline_mod
        from infinidev.engine.orchestration.staged_pipeline import (
            _goal_from_escalation,
        )
        from infinidev.engine.orchestration.task_renderer import render_task_xml
        from infinidev.engine.orchestration.task_schema import task_from_free_text
        from infinidev.prompts.flows import get_flow_config

        escalation = kwargs["escalation"]
        agent = kwargs["agent"]
        engine = kwargs["engine"]
        hooks = kwargs["hooks"]
        session_id = kwargs["session_id"]
        goal = _goal_from_escalation(escalation)

        node_objective = (getattr(node, "objective", "") or "").strip()
        is_goal_verification = getattr(node, "node_type", "") == "verification"
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
        )

        max_tool_calls = max(
            1,
            min(
                settings.REACT_MAX_TOOL_CALLS,
                int(budget.get("max_tool_calls", settings.REACT_MAX_TOOL_CALLS)),
            ),
        )
        hooks.on_phase("execute")
        hooks.on_status(
            "info",
            f"Graph node: {title} (budget {max_tool_calls} tool calls)",
        )
        agent.activate_context(session_id=session_id)
        try:
            result = engine.execute(
                agent=agent,
                task_prompt=task_prompt,
                verbose=True,
                initial_plan=None,
                initial_attachments=(
                    list(escalation.attachments) if escalation.attachments else None
                ),
                task=structured_task,
                max_iterations=settings.REACT_MAX_ITERATIONS,
                max_total_tool_calls=max_tool_calls,
                preserve_file_tracker=preserve_file_tracker,
                skip_plan=True,
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

        loop_status = getattr(engine, "_last_status", "") or "completed"
        if loop_status in {"blocked", "exhausted"}:
            return result, STATUS_BLOCKED
        if loop_status == "failed":
            return result, STATUS_FAILED

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
            rework_execute_kwargs={"skip_plan": True},
            run_verification=is_goal_verification,
        )
        review_status = getattr(engine, "_last_status", "") or "completed"
        if getattr(engine, "is_cancelled", False):
            return result, STATUS_CANCELLED
        if review_status in {"blocked", "exhausted"}:
            return result, STATUS_BLOCKED
        if review_status == "failed":
            return result, STATUS_FAILED
        return result, STATUS_COMPLETED

    # ── main loop ──────────────────────────────────────────────────────

    def run(self, **kwargs: Any) -> EngineResult:
        from infinidev.config.settings import settings

        escalation = kwargs.get("escalation")
        session_id = kwargs.get("session_id", "")
        run_id = kwargs.get("run_id") or f"graph_{id(self):x}"

        try:
            state = self._seed_state(run_id, session_id, escalation)
        except GraphInvariantError as exc:
            return EngineResult(
                engine_name=self.name,
                status=STATUS_BLOCKED,
                user_message=f"Could not seed the work graph: {exc}",
                summary="seed failed",
                engine=kwargs.get("engine"),
            )

        visits: dict[str, int] = {}
        leaf_runs = 0
        last_result = ""
        completed_results: list[tuple[str, str]] = []

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
                    transition_request=TransitionRequest(
                        target="staged",
                        reason="graph_leaf_budget_exhausted",
                    ),
                )

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
                )

            node_id = node.node_id
            try:
                state, _ = self._apply(
                    state, ActivateNodeOp(node_id=node_id, rationale=reason)
                )
                node_budget = {
                    "token_budget": settings.GRAPH_NODE_TOKEN_BUDGET,
                    "max_tool_calls": min(
                        settings.REACT_MAX_TOOL_CALLS,
                        settings.GRAPH_RUN_TOOL_BUDGET,
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
                    result_text, leaf_status = self._run_live_leaf(
                        capsule_text=capsule_text,
                        budget=node_budget,
                        node=node,
                        kwargs=kwargs,
                        # Each leaf owns its diff and review. Carrying the
                        # tracker across branches makes a read-only branch
                        # inherit earlier edits and lets review repair sibling
                        # failures inside the wrong node.
                        preserve_file_tracker=False,
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
                )

            if leaf_status != STATUS_COMPLETED:
                return EngineResult(
                    engine_name=self.name,
                    status=leaf_status,
                    user_message=result_text,
                    summary=f"Graph leaf {node_id} closed {leaf_status}.",
                    engine=kwargs.get("engine"),
                    state=state,
                    resume_token=session_id,
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
