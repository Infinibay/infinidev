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
        """Build the initial graph: one goal revision, a requirement and a
        work node that satisfies it."""
        request = getattr(escalation, "user_request", "") or ""
        title = next(
            (line.strip() for line in request.splitlines() if line.strip()),
            "Active goal",
        )[:120]

        state = GraphState(run_id=run_id, session_id=session_id)
        state, _ = self._apply(
            state,
            ReviseGoalOp(text=request, classification="new_requirement"),
        )

        requirement_id = new_node_id("req")
        work_id = new_node_id("work")
        patch = GraphPatchOp(
            add_nodes=[
                NodeSpec(
                    node_id=requirement_id,
                    node_type="requirement",
                    title=title,
                    objective=request,
                    expected_outcome="The user's request is satisfied.",
                ),
                NodeSpec(
                    node_id=work_id,
                    node_type="work",
                    title=f"Implement: {title}",
                    objective=request,
                    expected_outcome="The requested change is implemented and verified.",
                ),
            ],
            add_edges=[
                EdgeSpec(
                    source=requirement_id,
                    target=work_id,
                    edge_type=EDGE_DECOMPOSES_INTO,
                ),
                EdgeSpec(
                    source=work_id,
                    target=requirement_id,
                    edge_type=EDGE_SATISFIES,
                ),
            ],
            rationale="Seed graph from the escalated request.",
            based_on_revision=state.revision,
        )
        state, _ = self._apply(state, patch)
        return state

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
        from infinidev.engine.orchestration.task_schema import task_from_free_text
        from infinidev.prompts.flows import get_flow_config

        escalation = kwargs["escalation"]
        agent = kwargs["agent"]
        engine = kwargs["engine"]
        hooks = kwargs["hooks"]
        session_id = kwargs["session_id"]
        goal = _goal_from_escalation(escalation)

        approach = (
            '<approach authority="DERIVED">\n'
            "Execute only the active Graph node. Work incrementally, verify the "
            "outcome, and finish with step_complete when this node is satisfied. "
            "Do not invoke Stage planning or expand the user goal; the Graph "
            "scheduler retains orchestration authority.\n"
            "</approach>"
        )
        task_prompt = (
            f"{capsule_text}\n\n{approach}",
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

        literal_description = goal.user_request
        if len(literal_description.strip()) < 20:
            literal_description = f"User request (verbatim): {literal_description}"
        title = (getattr(node, "title", "") or "Graph node").strip()[:120]
        if len(title) < 5:
            title = f"{title} task"[:120]
        structured_task = task_from_free_text(
            literal_description,
            title=title,
            acceptance_criteria=list(goal.acceptance_criteria) or None,
            derived_verification_criteria=list(goal.derived_verification_criteria),
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

        result = pipeline_mod._run_review_phase(
            engine=engine,
            agent=agent,
            session_id=session_id,
            task_prompt=task_prompt,
            result=result,
            reviewer=kwargs.get("reviewer"),
            hooks=hooks,
            acceptance_criteria=list(goal.acceptance_criteria) or None,
            derived_verification_criteria=list(goal.derived_verification_criteria),
            task=structured_task,
            max_iterations=settings.REACT_MAX_ITERATIONS,
            max_total_tool_calls=max_tool_calls,
            rework_execute_kwargs={"skip_plan": True},
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

        while True:
            assessment = completion.evaluate_goal(state)
            if assessment.status == "complete":
                return EngineResult(
                    engine_name=self.name,
                    status=STATUS_COMPLETED,
                    user_message=last_result or "Goal completed.",
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
                        preserve_file_tracker=leaf_runs > 0,
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


__all__ = ["GraphEngineAdapter", "LeafExecutor"]
