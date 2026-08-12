"""Low-cost runtime behavior signals and bounded corrective interventions."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from typing import Any, Literal

from infinidev.engine.tool_executor import FILE_CHANGE_TOOLS


RuntimeBehaviorLabel = Literal[
    "command_timeout",
    "excessive_discovery",
    "premature_completion",
    "tool_schema_mismatch",
]

_DISCOVERY_TOOLS = frozenset(
    {
        "code_search",
        "describe_tool",
        "glob",
        "list_directory",
        "partial_read",
        "read_file",
        "search_symbols",
    }
)
_READ_TOOLS = frozenset({"partial_read", "read_file"})
_MODIFYING_KINDS = frozenset({"bugfix", "feature", "migration", "performance", "refactor"})
_INTERVENTIONS: dict[RuntimeBehaviorLabel, str] = {
    "command_timeout": (
        "<runtime-intervention>The last diagnostic timed out. Do not retry a similar "
        "probe. Use finite input or run the declared focused test.</runtime-intervention>"
    ),
    "excessive_discovery": (
        "<runtime-intervention>Discovery is sufficient for this narrow change. Use the "
        "loaded evidence, make the smallest scoped edit now, then run one focused test. "
        "Do not add another diagnostic probe unless that test exposes a new unknown."
        "</runtime-intervention>"
    ),
    "premature_completion": "",
    "tool_schema_mismatch": "",
}
_REASONING_INTERVENTIONS = {
    "excessive_exploration": (
        "<reasoning-intervention source=\"mini-model\">The visible reasoning is "
        "continuing broad exploration after locating the target. Act on the loaded "
        "evidence now: make the smallest reversible edit, then run its focused test."
        "</reasoning-intervention>"
    ),
    "retry_loop": (
        "<reasoning-intervention source=\"mini-model\">The visible reasoning proposes "
        "an equivalent retry after failure. Do not repeat it unchanged; alter cwd, input, "
        "arguments, or hypothesis using the failure evidence.</reasoning-intervention>"
    ),
    "premature_completion": (
        "<reasoning-intervention source=\"mini-model\">The visible reasoning is moving "
        "toward completion while required work remains. Continue the open requirement "
        "and close only with observable evidence.</reasoning-intervention>"
    ),
    "speculative_claim": (
        "<reasoning-intervention source=\"mini-model\">The visible reasoning promotes "
        "an unsupported explanation to fact. Keep it as a hypothesis, name the missing "
        "evidence, and use one focused observation to test it.</reasoning-intervention>"
    ),
    "verification_gap": (
        "<reasoning-intervention source=\"mini-model\">The visible reasoning would "
        "finish after an edit without relevant verification. Run the smallest check that "
        "covers the changed behavior before closing.</reasoning-intervention>"
    ),
}


@dataclass(frozen=True)
class RuntimeBehaviorSignal:
    """One auditable label derived only from observable execution state."""

    label: RuntimeBehaviorLabel
    source: str
    confidence: float
    step_index: int
    iteration: int
    tool_calls_seen: int
    intervention_queued: bool = False


def _tool_calls(messages: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for raw_call in message.get("tool_calls") or ():
            function = raw_call.get("function") or {}
            name = str(function.get("name") or raw_call.get("name") or "")
            raw_arguments = function.get("arguments") or raw_call.get("arguments") or {}
            if isinstance(raw_arguments, str):
                try:
                    raw_arguments = json.loads(raw_arguments)
                except (json.JSONDecodeError, TypeError):
                    raw_arguments = {}
            arguments = raw_arguments if isinstance(raw_arguments, dict) else {}
            calls.append((name, arguments))
    return calls


def _tool_results(messages: list[dict[str, Any]]) -> list[str]:
    return [
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == "tool"
    ]


def _has_observable_test_result(
    calls: list[tuple[str, dict[str, Any]]],
    results: list[str],
    state: Any,
) -> bool:
    from infinidev.engine.guidance.test_runners import (
        is_test_command,
        test_outcome_fingerprint,
    )

    return any(
        name == "execute_command"
        and is_test_command(json.dumps(arguments), state)
        and index < len(results)
        and test_outcome_fingerprint(results[index]) is not None
        for index, (name, arguments) in enumerate(calls)
    )


def task_allows_runtime_edit_nudge(task: Any | None) -> bool:
    """Return whether the task contract describes modifying repository work."""
    if task is None:
        return False
    profile = getattr(task, "task_profile", None)
    if profile is not None:
        authority = set(getattr(profile, "authority", ()) or ())
        operations = set(getattr(profile, "operations", ()) or ())
        return "modify" in authority and bool(operations & _MODIFYING_KINDS)
    return str(getattr(task, "kind", "")).casefold() in _MODIFYING_KINDS


def detect_runtime_behavior(
    messages: list[dict[str, Any]],
    state: Any,
    *,
    modifying_task: bool,
    step_index: int | None = None,
) -> tuple[RuntimeBehaviorSignal, ...]:
    """Detect conservative behavior patterns without an LLM or hidden reasoning."""
    calls = _tool_calls(messages)
    results = _tool_results(messages)
    if step_index is None:
        active_step = getattr(getattr(state, "plan", None), "active_step", None)
        step_index = int(
            getattr(active_step, "index", None)
            or getattr(state, "current_step_index", 0)
            or 0
        )
    iteration = int(getattr(state, "iteration_count", 0) or 0)
    signals: list[RuntimeBehaviorSignal] = []

    if any("Command timed out" in result for result in results):
        signals.append(
            RuntimeBehaviorSignal(
                label="command_timeout",
                source="tool-result:command-timeout",
                confidence=1.0,
                step_index=step_index,
                iteration=iteration,
                tool_calls_seen=len(calls),
            )
        )
    if any("wrong parameter name(s)" in result for result in results):
        signals.append(
            RuntimeBehaviorSignal(
                label="tool_schema_mismatch",
                source="tool-result:unexpected-kwargs",
                confidence=1.0,
                step_index=step_index,
                iteration=iteration,
                tool_calls_seen=len(calls),
            )
        )
    if any("Task completion advanced to the next planned Step" in result for result in results):
        signals.append(
            RuntimeBehaviorSignal(
                label="premature_completion",
                source="step-gate:pending-steps",
                confidence=1.0,
                step_index=step_index,
                iteration=iteration,
                tool_calls_seen=len(calls),
            )
        )

    names = [name for name, _ in calls]
    discovery_calls = sum(name in _DISCOVERY_TOOLS for name in names)
    read_calls = sum(name in _READ_TOOLS for name in names)
    wrote = any(name in FILE_CHANGE_TOOLS for name in names)
    tested = _has_observable_test_result(calls, results, state)
    if (
        modifying_task
        and len(calls) >= 8
        and discovery_calls >= 4
        and read_calls >= 2
        and not wrote
        and not tested
    ):
        signals.append(
            RuntimeBehaviorSignal(
                label="excessive_discovery",
                source="step-window:no-edit-no-test",
                confidence=0.95,
                step_index=step_index,
                iteration=iteration,
                tool_calls_seen=len(calls),
            )
        )
    return tuple(signals)


def observe_runtime_behavior(
    state: Any,
    messages: list[dict[str, Any]],
    *,
    task: Any | None,
    shadow_mode: bool,
    max_interventions: int,
    opened_files_budget_chars: int,
    step_index: int | None = None,
) -> str | None:
    """Record new signals and optionally queue one minimal intervention."""
    calls = _tool_calls(messages)
    results = _tool_results(messages)
    if any(name in FILE_CHANGE_TOOLS for name, _ in calls) or _has_observable_test_result(
        calls, results, state
    ):
        state.opened_files_prompt_max_chars = 0
    signals = detect_runtime_behavior(
        messages,
        state,
        modifying_task=task_allows_runtime_edit_nudge(task),
        step_index=step_index,
    )
    if not signals:
        return None

    seen = set(getattr(state, "runtime_behavior_seen", ()) or ())
    new_signals: list[RuntimeBehaviorSignal] = []
    for signal in signals:
        fingerprint = f"{signal.step_index}:{signal.label}"
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        new_signals.append(signal)
    state.runtime_behavior_seen = sorted(seen)
    if not new_signals:
        return None

    given = list(getattr(state, "runtime_interventions_given", ()) or ())
    queued_label: RuntimeBehaviorLabel | None = None
    if (
        not shadow_mode
        and not getattr(state, "pending_runtime_intervention", "")
        and len(given) < max(0, max_interventions)
    ):
        for signal in new_signals:
            intervention = _INTERVENTIONS[signal.label]
            if not intervention or signal.label in given:
                continue
            state.pending_runtime_intervention = intervention
            given.append(signal.label)
            state.runtime_interventions_given = given
            queued_label = signal.label
            if signal.label == "excessive_discovery":
                current = int(getattr(state, "opened_files_prompt_max_chars", 0) or 0)
                state.opened_files_prompt_max_chars = (
                    min(current, opened_files_budget_chars)
                    if current > 0
                    else opened_files_budget_chars
                )
            break

    events = list(getattr(state, "runtime_behavior_events", ()) or ())
    for signal in new_signals:
        payload = asdict(signal)
        payload["intervention_queued"] = signal.label == queued_label
        events.append(payload)
    state.runtime_behavior_events = events[-64:]
    return queued_label


def drain_runtime_intervention(state: Any) -> str:
    """Consume a queued intervention exactly once."""
    intervention = str(getattr(state, "pending_runtime_intervention", "") or "")
    state.pending_runtime_intervention = ""
    return intervention


def _current_tool_names(tool_calls: list[Any] | None) -> list[str]:
    names: list[str] = []
    for call in tool_calls or ():
        function = getattr(call, "function", None)
        name = getattr(call, "name", None) or getattr(function, "name", None)
        if name is None and isinstance(call, dict):
            name = call.get("name") or (call.get("function") or {}).get("name")
        if name:
            names.append(str(name))
    return names


def _failed_result_count(results: list[str]) -> int:
    count = 0
    for result in results:
        lowered = result.casefold()
        if "timed out" in lowered or "wrong parameter name" in lowered:
            count += 1
            continue
        try:
            payload = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        exit_code = payload.get("exit_code")
        if (
            payload.get("success") is False
            or isinstance(exit_code, int) and exit_code != 0
            or bool(payload.get("error"))
        ):
            count += 1
    return count


def reasoning_features(
    state: Any,
    messages: list[dict[str, Any]],
    *,
    task: Any | None,
    current_tool_calls: list[Any] | None,
) -> "ReasoningFeatures":
    """Build the observable half of the reasoning mini-model input."""
    from infinidev.engine.behavior.reasoning_classifier import ReasoningFeatures

    calls = _tool_calls(messages)
    names = [name for name, _ in calls]
    names.extend(_current_tool_names(current_tool_calls))
    results = _tool_results(messages)
    discovery_calls = sum(name in _DISCOVERY_TOOLS for name in names)
    edit_seen = bool(getattr(state, "task_has_edits", False)) or any(
        name in FILE_CHANGE_TOOLS for name in names
    )
    test_seen = bool(getattr(state, "last_test_output", "")) or _has_observable_test_result(
        calls, results, state
    )
    signatures = [
        json.dumps([name, arguments], sort_keys=True, default=str)
        for name, arguments in calls
    ]
    repeated = max(Counter(signatures).values(), default=0)
    plan = getattr(state, "plan", None)
    active = getattr(plan, "active_step", None)
    if plan is not None and hasattr(plan, "undischarged"):
        pending = bool(
            plan.undischarged(exclude_index=getattr(active, "index", None))
        )
    else:
        pending = any(
            getattr(step, "status", "") not in {"done", "blocked"}
            and step is not active
            for step in getattr(plan, "steps", ()) or ()
        )
    completion_attempt = "step_complete" in names
    evidence_seen = bool(calls or results or edit_seen or test_seen)
    return ReasoningFeatures(
        modifying_task=float(task_allows_runtime_edit_nudge(task)),
        discovery_pressure=min(1.0, discovery_calls / 4.0),
        edit_seen=float(edit_seen),
        test_seen=float(test_seen),
        failure_pressure=min(1.0, _failed_result_count(results) / 2.0),
        repeat_pressure=min(1.0, max(0, repeated - 1) / 2.0),
        required_work_pending=float(pending),
        completion_attempt=float(completion_attempt),
        evidence_seen=float(evidence_seen),
    )


def _reasoning_intervention_allowed(label: str, result: Any, features: Any) -> bool:
    if label == "excessive_exploration":
        return (
            features.modifying_task > 0
            and features.discovery_pressure >= 0.75
            and features.edit_seen == 0
            and features.test_seen == 0
        )
    if label == "retry_loop":
        return features.failure_pressure >= 0.5 and features.repeat_pressure >= 0.5
    if label == "premature_completion":
        return features.required_work_pending > 0 and features.completion_attempt > 0
    if label == "verification_gap":
        return (
            features.edit_seen > 0
            and features.test_seen == 0
            and features.completion_attempt > 0
        )
    if label == "speculative_claim":
        # A model's opening hypothesis is not itself a failure. Until natural
        # trajectories support a streak-based gate, only an unusually strong
        # semantic activation can interrupt before any evidence exists.
        return features.evidence_seen == 0 and result.score - result.threshold >= 0.30
    return False


def observe_reasoning_behavior(
    state: Any,
    reasoning_text: str,
    messages: list[dict[str, Any]],
    *,
    task: Any | None,
    current_tool_calls: list[Any] | None,
    sources: tuple[str, ...] = (),
    shadow_mode: bool,
    max_interventions: int,
) -> dict[str, object] | None:
    """Classify provider-visible reasoning and queue one evidence-gated prompt."""
    if not reasoning_text.strip():
        return None
    from infinidev.engine.behavior.reasoning_classifier import classify_reasoning

    features = reasoning_features(
        state,
        messages,
        task=task,
        current_tool_calls=current_tool_calls,
    )
    result = classify_reasoning(reasoning_text, features)
    label = result.label or "uncategorized"
    intervention_queued = False
    given = list(getattr(state, "runtime_interventions_given", ()) or ())
    intervention_key = f"reasoning:{label}"
    if (
        result.label in _REASONING_INTERVENTIONS
        and _reasoning_intervention_allowed(result.label, result, features)
        and not shadow_mode
        and not getattr(state, "pending_runtime_intervention", "")
        and intervention_key not in given
        and len(given) < max(0, max_interventions)
    ):
        state.pending_runtime_intervention = _REASONING_INTERVENTIONS[result.label]
        given.append(intervention_key)
        state.runtime_interventions_given = given
        intervention_queued = True

    plan = getattr(state, "plan", None)
    active = getattr(plan, "active_step", None)
    payload: dict[str, object] = {
        "label": f"reasoning:{label}",
        "source": "static-qwen3-reasoning-mini-head",
        "confidence": result.score,
        "threshold": result.threshold,
        "runner_up_margin": result.runner_up_margin,
        "step_index": int(getattr(active, "index", 0) or 0),
        "iteration": int(getattr(state, "iteration_count", 0) or 0),
        "tool_calls_seen": len(_tool_calls(messages)),
        "intervention_queued": intervention_queued,
        "space_id": result.space_id,
        "classifier_version": result.classifier_version,
        "abstention_reason": result.abstention_reason,
        "reasoning_sources": list(sources),
        "features": {
            name: float(getattr(features, name))
            for name in features.__dataclass_fields__
        },
    }
    events = list(getattr(state, "runtime_behavior_events", ()) or ())
    events.append(payload)
    state.runtime_behavior_events = events[-64:]
    return payload


def observe_semantic_behavior(state: Any) -> dict[str, object] | None:
    """Classify the newest completed step in shadow mode using static Qwen3."""
    history = list(getattr(state, "history", ()) or ())
    if not history:
        return None
    record = history[-1]
    fingerprint = f"{record.step_index}:semantic"
    seen = set(getattr(state, "runtime_behavior_seen", ()) or ())
    if fingerprint in seen:
        return None
    seen.add(fingerprint)
    state.runtime_behavior_seen = sorted(seen)

    from infinidev.engine.behavior.semantic_classifier import classify_step_behavior

    text = "\n".join(
        part
        for part in (
            record.summary,
            record.changes_made,
            record.discovered_context,
            record.pending_items,
            record.anti_patterns,
            " ".join(record.behavior_bad),
        )
        if part
    )
    result = classify_step_behavior(text)
    payload: dict[str, object] = {
        "label": f"semantic:{result.label}" if result.label else "semantic:uncategorized",
        "source": "static-qwen3-step-summary",
        "confidence": result.score,
        "runner_up_margin": result.runner_up_margin,
        "neutral_margin": result.neutral_margin,
        "step_index": record.step_index,
        "iteration": int(getattr(state, "iteration_count", 0) or 0),
        "tool_calls_seen": record.tool_calls_count,
        "intervention_queued": False,
        "space_id": result.space_id,
        "classifier_version": result.classifier_version,
        "abstention_reason": result.abstention_reason,
    }
    events = list(getattr(state, "runtime_behavior_events", ()) or ())
    events.append(payload)
    state.runtime_behavior_events = events[-64:]
    return payload


__all__ = [
    "RuntimeBehaviorSignal",
    "detect_runtime_behavior",
    "drain_runtime_intervention",
    "observe_reasoning_behavior",
    "observe_runtime_behavior",
    "observe_semantic_behavior",
    "reasoning_features",
    "task_allows_runtime_edit_nudge",
]
