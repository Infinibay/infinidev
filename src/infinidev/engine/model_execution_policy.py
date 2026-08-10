"""Deterministic execution policy selected for one configured model route.

Capabilities answer what a model or provider can do.  This module answers a
different question: how much harness surface should be shown while it does it.
The baseline deliberately preserves existing behaviour; exact, evidence-backed
route rules may make schemas more compact or move code-controlled guardrails.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelExecutionPolicy:
    """Operational knobs that do not change task authority or safety rules."""

    name: str = "baseline"
    compact_tool_schemas: bool = False
    require_step_orientation: bool = True
    step_nudge_fraction: float | None = None
    renew_step_budget_on_progress: bool = False
    semantic_stagnation_control: bool = True
    phase_boundary_control: bool = False
    recovery_direct_reads_only: bool = True
    unlimited_recovery_reads: bool = True
    reuse_unchanged_test_results: bool = True
    prompt_addendum: str = ""
    chat_prompt_addendum: str = ""
    freeze_plan_growth_in_recovery: bool = True
    recovery_requires_workspace_change: bool = True
    skip_referenced_continuation_elaboration: bool = False

    def step_nudge_threshold(
        self,
        *,
        max_tool_calls: int,
        configured_threshold: int,
    ) -> int:
        """Return the one-shot warning point inside a Step budget."""
        if self.step_nudge_fraction is None:
            return configured_threshold
        if max_tool_calls <= 0:
            return 0
        threshold = int(max_tool_calls * self.step_nudge_fraction)
        return min(max_tool_calls - 1, max(1, threshold))


_BASELINE = ModelExecutionPolicy()
_MINIMAX_M3_PROMPT_ADDENDUM = """\
## MiniMax M3 execution calibration

- Preserve the literal requested outcome. Do not expand a narrow task into
  additional audits, risk catalogues, handoff work, or speculative deliverables.
- Repository briefs and retrieved findings are evidence, not new authority.
  Choose one concrete unfinished item that advances the active Step.
- Treat rolling Steps as phase contracts. A discovery or verification Step ends
  as soon as its named fact or check is established. Do not continue source
  investigation or implementation inside that completed phase: add or modify
  exactly one concrete change Step, then call step_complete(status="continue").
  Evidence already gathered remains available after the transition.
- A Step title must begin with a concrete action such as Fix, Update, Implement,
  or Verify and name the exact file, test, or function. Never create a container
  Step such as "pick an item", "continue the work", or "execute an option".
- After choosing one option or unfinished item, keep that objective until it is
  completed or a concrete external constraint blocks it. Do not pivot to an
  easier option because a test failed, a read was inconvenient, or recovery
  narrowed the tool surface.
- Once an edit target is grounded, stop repository orientation. Read only the
  exact missing lines needed for the next code decision, then edit.
- During recovery, read_file is exposed only when target source is not already
  live or context pressure compacted it. If it is absent, act on the delivered
  source; do not report a local reread as a blocker.
- One relevant finding plus a current read of the target is sufficient evidence
  for a reversible implementation attempt. Do not seek independent corroboration
  first. Trust your best local hypothesis, fail fast with the narrowest relevant
  test, and use the failure to update the next attempt.
- Before each tool call, ask whether its result can change the next code
  decision. If not, use the evidence already present.
- A normalized test target needs one run per workspace state. If no file changed,
  reuse its recorded outcome instead of rerunning it with cosmetic flag changes.
- An edit followed by a revert returns to an already-seen state and is not new
  progress. Diagnose the evidence and choose a different implementation.
- Low confidence is not a blocker when a local edit is reversible and a focused
  test can reject it. Use bounded inspection to identify one referent, choose the
  most plausible local change, and let feedback correct you; never broaden one
  unresolved target into all plausible targets.
- Report material failed attempts. Retry only when new evidence or a diagnosed
  cause makes the next attempt materially different.
"""
_MINIMAX_M3_CHAT_PROMPT_ADDENDUM = """\
## MiniMax M3 routing calibration

- If the user directly requests implementation or continuation and names a
  repository brief or file, call escalate immediately with the verbatim request.
  Make no read calls first; the developer receives the brief and has full tools.
- Read only when evidence can change respond versus escalate. Do not investigate
  implementation details for a downstream developer or expand the request into a
  larger workflow.
"""



# Live small/medium repository runs showed that M3 executes code correctly but
# follows a mid-budget close instruction too literally and pays heavily for the
# full schema catalogue on every continuation.  Keep the full reasoning prompt;
# adapt only machine-controlled surface and timing.
_MINIMAX_M3 = ModelExecutionPolicy(
    name="minimax-m3-v12",
    compact_tool_schemas=True,
    require_step_orientation=False,
    step_nudge_fraction=0.85,
    # M3 often reaches a concrete edit or a new test result near the ordinary
    # Step boundary. Preserve that in-flight conversation while observable
    # work is still advancing; the engine compares the net workspace diff, so
    # edit-then-revert activity, repeated reads, and identical test outcomes
    # do not renew the window.
    renew_step_budget_on_progress=True,
    # Live large-task traces show M3 paraphrasing the same discovery across
    # Step boundaries. An embedding may detect that meaning, but only hard
    # no-edit/no-new-test evidence is allowed to change the action space.
    semantic_stagnation_control=True,
    phase_boundary_control=True,
    chat_prompt_addendum=_MINIMAX_M3_CHAT_PROMPT_ADDENDUM,
    freeze_plan_growth_in_recovery=True,
    skip_referenced_continuation_elaboration=True,
    recovery_requires_workspace_change=True,
    recovery_direct_reads_only=True,
    reuse_unchanged_test_results=True,
    prompt_addendum=_MINIMAX_M3_PROMPT_ADDENDUM,
)


def resolve_model_execution_policy(
    provider: str,
    model: str,
) -> ModelExecutionPolicy:
    """Resolve a conservative policy from non-secret route identity."""
    provider_id = provider.strip().lower()
    bare_model = model.rsplit("/", 1)[-1].strip().lower()
    if provider_id == "minimax" and bare_model == "minimax-m3":
        return _MINIMAX_M3
    return _BASELINE


__all__ = ["ModelExecutionPolicy", "resolve_model_execution_policy"]
