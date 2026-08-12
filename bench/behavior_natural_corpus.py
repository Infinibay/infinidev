"""Extract privacy-reduced behavior windows from real agent run artifacts.

The extractor never stores tool-result bodies, source excerpts, or ``think``
arguments. Conservative automatic labels are kept separate from ambiguous
windows that require review.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from infinidev.engine.guidance.test_runners import test_outcome_fingerprint
from infinidev.engine.tool_executor import FILE_CHANGE_TOOLS


SCHEMA_VERSION = 1
EXTRACTOR_VERSION = "natural-observable-windows-v1"
_EXCLUDED_TRACE_TOOLS = frozenset({"think"})
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
_TEST_COMMAND_RE = re.compile(
    r"(?:^|\s)(?:uv\s+run\s+)?"
    r"(?:pytest|cargo\s+test|npm\s+test|npx\s+ava|pnpm\s+test|yarn\s+test)"
    r"(?:\s|$)",
    re.IGNORECASE,
)
_SECRET_RE = re.compile(
    r"(?i)(?:pypi-[A-Za-z0-9_-]{16,}|sk-[A-Za-z0-9_-]{16,}|"
    r"(?:api[_-]?key|token|secret)\s*[:=]\s*[^\s,;]+)"
)
_TEMP_WORKSPACE_RE = re.compile(r"/tmp/infinidev-agent-task-[^/\s]+/repo")
_HOME_PATH_RE = re.compile(r"/home/[^/\s]+/(?:[^\s,;:`]+)")
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)


@dataclass(frozen=True)
class NaturalBehaviorWindow:
    """One natural, replay-addressable, observable behavior example."""

    id: str
    source_artifact: str
    source_sha256: str
    task_id: str
    project_family: str
    task_category: str
    provider: str
    model: str
    model_identity: str
    condition: str
    repetition: int
    window_kind: str
    step_ordinal: int
    step_index: int
    text: str
    label: str | None
    label_source: str
    review_status: str
    hard_negative_for: tuple[str, ...]
    trace_aligned: bool
    tool_calls: int
    discovery_calls: int
    read_calls: int
    edit_calls: int
    test_calls: int
    failed_calls: int
    repeated_call_max: int
    net_workspace_changed: bool
    successful_edit_count: int
    run_success: bool
    extractor_version: str = EXTRACTOR_VERSION
    schema_version: int = SCHEMA_VERSION


def sanitize_text(value: object, *, max_chars: int = 2400) -> str:
    """Remove secrets and host-specific paths from model-visible prose."""
    text = str(value or "")
    text = _SECRET_RE.sub("<secret>", text)
    text = _TEMP_WORKSPACE_RE.sub("<workspace>", text)
    text = _HOME_PATH_RE.sub("<workspace-path>", text)
    text = _EMAIL_RE.sub("<email>", text)
    text = " ".join(text.split())
    return text[:max_chars]


def _artifact_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_artifact(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _is_test_trace(item: dict[str, Any], verify_command: str = "") -> bool:
    if item.get("tool_name") != "execute_command":
        return False
    command = str((item.get("arguments") or {}).get("command") or "")
    declared_verify = bool(verify_command and verify_command in command)
    if not declared_verify and not _TEST_COMMAND_RE.search(command):
        return False
    result = str(item.get("result") or "")
    if test_outcome_fingerprint(result) is not None:
        return True
    if declared_verify:
        try:
            payload = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return False
        return payload.get("exit_code") == 0 or payload.get("success") is True
    return False


def _signature(item: dict[str, Any]) -> str:
    arguments = item.get("arguments") or {}
    safe_arguments = {
        key: sanitize_text(value, max_chars=160)
        for key, value in arguments.items()
        if key in {
            "command",
            "context",
            "file_path",
            "index",
            "limit",
            "offset",
            "query",
            "start_line",
            "symbol",
            "tool_name",
        }
    }
    return json.dumps(
        [str(item.get("tool_name") or ""), safe_arguments],
        ensure_ascii=False,
        sort_keys=True,
    )


def _trace_line(index: int, item: dict[str, Any], verify_command: str) -> str:
    name = str(item.get("tool_name") or "unknown")
    arguments = item.get("arguments") or {}
    detail = ""
    for key in ("file_path", "command", "query", "context", "tool_name"):
        if arguments.get(key):
            detail = f" {key}={sanitize_text(arguments[key], max_chars=140)}"
            break
    result = str(item.get("result") or "")
    if "timed out" in result.casefold():
        outcome = "timeout"
    elif _trace_failed(item):
        outcome = "failed"
    elif _is_test_trace(item, verify_command):
        outcome = "test-result"
    else:
        outcome = "ok"
    return f"{index}. {name}{detail} -> {outcome}"


def _trace_text(task: dict[str, Any], trace: list[dict[str, Any]]) -> str:
    verify_command = str(task.get("verify_command") or "")
    lines = [
        f"Task category: {sanitize_text(task.get('category'), max_chars=80)}.",
        f"User request: {sanitize_text(task.get('request'), max_chars=500)}",
        "Observable action sequence:",
    ]
    visible = trace[-16:]
    offset = len(trace) - len(visible)
    lines.extend(
        _trace_line(offset + index + 1, item, verify_command)
        for index, item in enumerate(visible)
    )
    return "\n".join(lines)


def _features(trace: list[dict[str, Any]], *, verify_command: str = "") -> dict[str, int]:
    names = [str(item.get("tool_name") or "") for item in trace]
    repeats = Counter(_signature(item) for item in trace)
    return {
        "tool_calls": len(trace),
        "discovery_calls": sum(name in _DISCOVERY_TOOLS for name in names),
        "read_calls": sum(name in _READ_TOOLS for name in names),
        "edit_calls": sum(name in FILE_CHANGE_TOOLS for name in names),
        "test_calls": sum(_is_test_trace(item, verify_command) for item in trace),
        "failed_calls": sum(_trace_failed(item) for item in trace),
        "repeated_call_max": max(repeats.values(), default=0),
    }


def _trace_failed(item: dict[str, Any]) -> bool:
    """Recognize failures even when a hook did not set its coarse flag."""
    if item.get("failed"):
        return True
    try:
        payload = json.loads(str(item.get("result") or ""))
    except (json.JSONDecodeError, TypeError):
        return False
    exit_code = payload.get("exit_code") if isinstance(payload, dict) else None
    return (isinstance(exit_code, int) and exit_code != 0) or (
        isinstance(payload, dict) and payload.get("success") is False
    )


def _task_modifies_files(task: dict[str, Any]) -> bool:
    return str(task.get("category") or "") in {
        "bugfix", "feature", "implementation", "migration", "performance", "refactor"
    }


def _runtime_labels(artifact: dict[str, Any]) -> dict[int, set[str]]:
    by_step: dict[int, set[str]] = defaultdict(set)
    for event in artifact.get("runtime_behavior_events", ()):
        label = str(event.get("label") or "")
        if not label or label.startswith("semantic:"):
            continue
        normalized = {
            "excessive_discovery": "excessive_exploration",
        }.get(label, label)
        by_step[int(event.get("step_index") or 0)].add(normalized)
    return by_step


def _label_window(
    *,
    record: dict[str, Any],
    features: dict[str, int],
    runtime_labels: set[str],
    modifying_task: bool,
    run_success: bool,
    trace_aligned: bool,
) -> tuple[str | None, str, str, tuple[str, ...]]:
    supported_runtime = runtime_labels & {
        "command_timeout", "excessive_exploration", "premature_completion", "tool_schema_mismatch"
    }
    if len(supported_runtime) == 1:
        return next(iter(supported_runtime)), "runtime_event", "approved", ()
    if len(supported_runtime) > 1:
        return None, "conflicting_runtime_events", "needs_review", ()
    if trace_aligned and features["repeated_call_max"] >= 3 and features["failed_calls"]:
        return "retry_loop", "deterministic_trace", "approved", ()
    if (
        trace_aligned
        and modifying_task
        and features["tool_calls"] >= 8
        and features["discovery_calls"] >= 4
        and features["read_calls"] >= 2
        and features["edit_calls"] == 0
        and features["test_calls"] == 0
    ):
        return "excessive_exploration", "deterministic_trace", "approved", ()

    edits = int(record.get("successful_edit_count") or 0)
    changed = bool(record.get("net_workspace_changed"))
    test_evidence = features["test_calls"] > 0 or bool(record.get("test_outcome_fingerprints"))
    if run_success and (edits > 0 or changed) and test_evidence:
        return "healthy_progress", "observable_progress", "approved", ()
    if not modifying_task and run_success and not changed and edits == 0:
        hard_for = (
            ("excessive_exploration",)
            if features["discovery_calls"] >= 4 or features["read_calls"] >= 2
            else ()
        )
        return None, "legitimate_non_modifying_task", "approved", hard_for
    if trace_aligned and features["test_calls"] == 1 and features["edit_calls"] == 0:
        return None, "single_diagnostic_test", "approved", ("retry_loop", "verification_gap")
    return None, "insufficient_observable_evidence", "needs_review", ()


def _window_text(
    task: dict[str, Any], record: dict[str, Any], features: dict[str, int]
) -> str:
    parts = [
        f"Task category: {sanitize_text(task.get('category'), max_chars=80)}.",
        f"User request: {sanitize_text(task.get('request'), max_chars=500)}",
        f"Step summary: {sanitize_text(record.get('summary'))}",
    ]
    for label, key in (
        ("Changes", "changes_made"),
        ("Discovered", "discovered_context"),
        ("Pending", "pending_items"),
        ("Anti-patterns", "anti_patterns"),
    ):
        value = sanitize_text(record.get(key), max_chars=600)
        if value:
            parts.append(f"{label}: {value}")
    parts.append(
        "Observable counts: "
        + ", ".join(f"{key}={value}" for key, value in sorted(features.items()))
        + "."
    )
    return "\n".join(parts)


def extract_artifact(path: Path, *, root: Path) -> list[NaturalBehaviorWindow]:
    """Extract conservative per-Step windows from one run artifact."""
    artifact = json.loads(path.read_text(encoding="utf-8"))
    task = artifact.get("task") or {}
    records = list(artifact.get("action_records") or ())
    trace = [
        item for item in artifact.get("tool_trace") or ()
        if item.get("tool_name") not in _EXCLUDED_TRACE_TOOLS
    ]
    expected_calls = sum(int(record.get("tool_calls_count") or 0) for record in records)
    aligned = expected_calls == len(trace)
    runtime_by_step = _runtime_labels(artifact)
    step_counts = Counter(
        int(record.get("step_index") or index)
        for index, record in enumerate(records)
    )
    source_sha = _artifact_sha256(path)
    project_family = str(
        task.get("repository_fixture") or task.get("family") or task.get("id") or "unknown"
    )
    modifying_task = _task_modifies_files(task)
    run_success = bool(
        artifact.get("verify_exit_code") == 0 and not artifact.get("error")
    )
    offset = 0
    windows: list[NaturalBehaviorWindow] = []
    for ordinal, record in enumerate(records):
        count = int(record.get("tool_calls_count") or 0)
        record_trace = trace[offset : offset + count] if aligned else []
        offset += count
        features = _features(
            record_trace,
            verify_command=str(task.get("verify_command") or ""),
        )
        step_index = int(record.get("step_index") or ordinal)
        label, label_source, review_status, hard_negative_for = _label_window(
            record=record,
            features=features,
            runtime_labels=(
                runtime_by_step.get(step_index, set())
                if step_counts[step_index] == 1
                else set()
            ),
            modifying_task=modifying_task,
            run_success=run_success,
            trace_aligned=aligned,
        )
        identity = f"{source_sha}:{ordinal}:{step_index}"
        windows.append(NaturalBehaviorWindow(
            id=hashlib.sha256(identity.encode()).hexdigest()[:20],
            source_artifact=_relative_artifact(path, root),
            source_sha256=source_sha,
            task_id=str(task.get("id") or path.parent.name),
            project_family=project_family,
            task_category=str(task.get("category") or "unknown"),
            provider=str(artifact.get("provider") or "unknown"),
            model=str(artifact.get("model") or "unknown"),
            model_identity=str(artifact.get("model_identity") or "unknown"),
            condition=str(artifact.get("condition") or "unknown"),
            repetition=int(artifact.get("repetition") or 0),
            window_kind="step_summary",
            step_ordinal=ordinal,
            step_index=step_index,
            text=_window_text(task, record, features),
            label=label,
            label_source=label_source,
            review_status=review_status,
            hard_negative_for=hard_negative_for,
            trace_aligned=aligned,
            net_workspace_changed=bool(record.get("net_workspace_changed")),
            successful_edit_count=int(record.get("successful_edit_count") or 0),
            run_success=run_success,
            **features,
        ))
    windows.extend(_extract_trace_prefixes(
        artifact,
        task=task,
        trace=trace,
        path=path,
        root=root,
        source_sha=source_sha,
        project_family=project_family,
        modifying_task=modifying_task,
        run_success=run_success,
    ))
    return windows


def _extract_trace_prefixes(
    artifact: dict[str, Any],
    *,
    task: dict[str, Any],
    trace: list[dict[str, Any]],
    path: Path,
    root: Path,
    source_sha: str,
    project_family: str,
    modifying_task: bool,
    run_success: bool,
) -> list[NaturalBehaviorWindow]:
    """Extract the earliest hard-evidence prefix for each supported label."""
    verify_command = str(task.get("verify_command") or "")
    candidates: list[tuple[str | None, int, str, tuple[str, ...]]] = []
    emitted: set[str | None] = set()
    repeatable = {"describe_tool", "execute_command", "partial_read", "read_file"}
    for end in range(1, len(trace) + 1):
        prefix = trace[:end]
        features = _features(prefix, verify_command=verify_command)
        latest = prefix[-1]
        result = str(latest.get("result") or "")
        labels: list[str] = []
        if "timed out" in result.casefold():
            labels.append("command_timeout")
        if "wrong parameter name(s)" in result or "unexpected kwargs" in result:
            labels.append("tool_schema_mismatch")
        repeated = Counter(
            _signature(item)
            for item in prefix
            if str(item.get("tool_name") or "") in repeatable
        )
        if max(repeated.values(), default=0) >= 3:
            labels.append("retry_loop")
        if (
            modifying_task
            and features["tool_calls"] >= 8
            and features["discovery_calls"] >= 4
            and features["read_calls"] >= 2
            and features["edit_calls"] == 0
            and features["test_calls"] == 0
        ):
            labels.append("excessive_exploration")
        for label in labels:
            if label in emitted:
                continue
            emitted.add(label)
            candidates.append((label, end, "deterministic_trace_prefix", ()))

    final_features = _features(trace, verify_command=verify_command)
    if (
        run_success
        and modifying_task
        and final_features["edit_calls"] > 0
    ):
        failed_positions = [
            index for index, item in enumerate(trace) if _trace_failed(item)
        ]
        adaptive_recovery = False
        if failed_positions:
            first_failure = failed_positions[0]
            edit_positions = [
                index
                for index, item in enumerate(trace)
                if index > first_failure and item.get("tool_name") in FILE_CHANGE_TOOLS
            ]
            adaptive_recovery = any(
                index > edit_positions[0]
                and item.get("tool_name") == "execute_command"
                and not _trace_failed(item)
                for index, item in enumerate(trace)
            ) if edit_positions else False
        candidates.append((
            "healthy_progress",
            len(trace),
            "verified_run_outcome",
            ("retry_loop",) if adaptive_recovery else (),
        ))
    elif run_success and not modifying_task and final_features["edit_calls"] == 0:
        snapshot_ends: list[int] = []
        for evidence_target in (4, 8):
            for end in range(1, len(trace) + 1):
                prefix_features = _features(trace[:end], verify_command=verify_command)
                if prefix_features["discovery_calls"] >= evidence_target:
                    snapshot_ends.append(end)
                    break
        if final_features["discovery_calls"] >= 4:
            snapshot_ends.append(len(trace))
        for end in dict.fromkeys(snapshot_ends):
            candidates.append((
                None,
                end,
                "legitimate_discovery_prefix",
                ("excessive_exploration",),
            ))

    result_windows: list[NaturalBehaviorWindow] = []
    for ordinal, (label, end, source, hard_negative_for) in enumerate(candidates):
        prefix = trace[:end]
        features = _features(prefix, verify_command=verify_command)
        identity = f"{source_sha}:trace:{ordinal}:{label}:{end}"
        result_windows.append(NaturalBehaviorWindow(
            id=hashlib.sha256(identity.encode()).hexdigest()[:20],
            source_artifact=_relative_artifact(path, root),
            source_sha256=source_sha,
            task_id=str(task.get("id") or path.parent.name),
            project_family=project_family,
            task_category=str(task.get("category") or "unknown"),
            provider=str(artifact.get("provider") or "unknown"),
            model=str(artifact.get("model") or "unknown"),
            model_identity=str(artifact.get("model_identity") or "unknown"),
            condition=str(artifact.get("condition") or "unknown"),
            repetition=int(artifact.get("repetition") or 0),
            window_kind="tool_prefix",
            step_ordinal=ordinal,
            step_index=-1,
            text=_trace_text(task, prefix),
            label=label,
            label_source=source,
            review_status="approved",
            hard_negative_for=hard_negative_for,
            trace_aligned=True,
            net_workspace_changed=features["edit_calls"] > 0,
            successful_edit_count=features["edit_calls"],
            run_success=run_success,
            **features,
        ))
    return result_windows


def discover_artifacts(roots: Iterable[Path]) -> list[Path]:
    """Return unique run artifacts in deterministic order."""
    found: set[Path] = set()
    for root in roots:
        if root.is_file() and root.name == "run.json":
            found.add(root)
        elif root.is_dir():
            found.update(root.rglob("run.json"))
    return sorted(found)


def write_corpus(
    roots: list[Path], output: Path, review_output: Path, *, repository_root: Path
) -> dict[str, Any]:
    """Extract artifacts and write approved/review JSONL outputs."""
    windows = [
        window
        for artifact in discover_artifacts(roots)
        for window in extract_artifact(artifact, root=repository_root)
    ]
    approved = [window for window in windows if window.review_status == "approved"]
    review = [window for window in windows if window.review_status != "approved"]
    for path, values in ((output, approved), (review_output, review)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(
                json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) + "\n"
                for item in values
            ),
            encoding="utf-8",
        )
    return {
        "artifacts": len(discover_artifacts(roots)),
        "windows": len(windows),
        "approved": len(approved),
        "needs_review": len(review),
        "trace_aligned": sum(window.trace_aligned for window in windows),
        "labels": dict(sorted(Counter(
            window.label or "uncategorized" for window in approved
        ).items())),
        "hard_negatives": sum(bool(window.hard_negative_for) for window in approved),
        "project_families": len({window.project_family for window in windows}),
        "models": len({(window.provider, window.model) for window in windows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(
        write_corpus(
            args.roots,
            args.output,
            args.review_output,
            repository_root=args.repository_root,
        ),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
