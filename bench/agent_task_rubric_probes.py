"""Decide the rubric items that can be decided from the artifact alone.

Half of the ``human_review`` rubric asks a question a program can answer: does
the diff touch the stage that was wrong or does it compensate in the pipeline;
does the added money arithmetic use integer division; did the run re-invoke the
command it already saw fail. Those have answers in ``run.json`` and no reason to
depend on a judge's opinion — and a judge who shipped the change under review is
exactly the wrong instrument for them.

Two rules make the results trustworthy:

* **A probe that cannot see abstains.** Every probe returns ``(score, evidence)``
  with ``score`` set to ``None`` when the artifact does not carry what the probe
  needs. An abstention is reported as an abstention; it is never folded in as a
  zero.
* **A trace is not a record of the work when the work was delegated.** In
  ``orchestrator`` mode ``tool_trace`` holds the *principal's* calls, and the
  worker's own calls — the reads, the edit, the test run — are not in it, so
  criteria that the rubric phrases as "the exact tool trace shows …" cannot be
  judged for those runs. The diff is complete for both modes; the trace is not.
  Trace-dependent probes abstain when any team tool appears.

Usage::

    python -m bench.agent_task_rubric_probes bench/runs/<date>-<name> \\
        --markdown bench/runs/<date>-<name>/probes.md
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable

#: ``(score or None, evidence)``. ``None`` means the probe could not see.
Outcome = tuple["int | None", str]
Probe = Callable[[dict[str, Any]], Outcome]

#: Any of these in the trace means real work happened in a worker, and the
#: trace no longer accounts for it.
_TEAM_TOOLS = frozenset({
    "team_delegate", "team_create_ticket", "team_idle", "team_wait",
    "team_review_ticket", "team_tool_catalog", "team_read", "team_write_note",
})


# ── artifact accessors ──────────────────────────────────────────────────

def _added_lines(diff: str) -> list[str]:
    return [
        line[1:] for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]


def _removed_lines(diff: str) -> list[str]:
    return [
        line[1:] for line in diff.splitlines()
        if line.startswith("-") and not line.startswith("---")
    ]


def _changed_files(diff: str) -> list[str]:
    """Source files the diff covers, without compiled or cached artifacts.

    A run that imports a module recompiles its ``.pyc``; if the fixture shipped
    one — the fixtures carry no ``.gitignore``, so a stray ``__pycache__`` from
    someone running pytest inside the fixture became a tracked baseline file —
    the diff is mostly binary cache noise. ``wide-sum`` once produced 7 980
    characters of diff of which 7 370 were ``.pyc``. Judging "the fix landed in
    the stage" against that list would fail a correct fix.
    """

    files = re.findall(r"^### (.+?) \([^)]*\)$", diff, re.MULTILINE)
    return [
        path for path in files
        if not any(
            part in _IGNORED_PARTS or part.endswith((".pyc", ".pyo"))
            for part in path.split("/")
        )
    ]


#: Directory and file names that are build output, not a reviewer's business.
_IGNORED_PARTS = frozenset({
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache",
    ".tox", ".nox", ".venv", "venv", ".git",
})


def _tool_calls(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    return list(artifact.get("tool_trace") or [])


def _arguments_text(call: dict[str, Any]) -> str:
    args = call.get("arguments")
    if isinstance(args, dict):
        return " ".join(f"{k}={v}" for k, v in args.items())
    return str(args)


def _answer(artifact: dict[str, Any]) -> str:
    return str(artifact.get("final_answer") or "")


def _delegated(artifact: dict[str, Any]) -> bool:
    """Whether this run delegated work the trace does not account for."""

    return any(str(call.get("tool_name") or "") in _TEAM_TOOLS
               for call in _tool_calls(artifact))


_DELEGATED_REASON = (
    "the trace holds only the principal's calls; the worker's reads, edit and "
    "test run are not in the artifact, so this criterion is not judgeable here"
)


_COMMAND_IN_ANSWER = re.compile(
    r"\b(?:pytest|python3?\s+-m|python3?\s+\S+\.py|npm|make|cargo|go test)\b"
)
_RESULT_IN_ANSWER = re.compile(
    # "exit 0", "exited 0", "exits 0", "exit code 0", "exit status: 0",
    # "exit code = 0" — the plural and colon forms are what models write.
    r"\b(?:passed|passing|exit(?:ed|s)?(?:\s+(?:code|status))?\s*[:=]?\s*0"
    r"|→\s*0|rc\s*0|VERIFY_OK)\b",
    re.IGNORECASE,
)
_REMAINING_IN_ANSWER = re.compile(
    # A handoff points at what is still open. The plural and the passive are
    # how that is actually written — "Open decisions now owned by the user"
    # was missed by a pattern that required the singular with a word boundary.
    r"\b(?:not done|remains?|remaining|belongs? to you|belongs? to the user|"
    r"left to you|deferred|open (?:decision|question)s?"
    r"|still (?:need|needs|open|pending|outstanding)"
    r"|blocked on|outstanding|pending (?:the )?user"
    r"|owned by the user|needs? (?:your|user|product|security) "
    r"(?:confirmation|input|answer|decision))\b",
    re.IGNORECASE,
)


def _commands(artifact: dict[str, Any]) -> list[str]:
    """Shell commands the trace actually recorded, with their arguments."""

    return [
        str((call.get("arguments") or {}).get("command") or "")
        for call in _tool_calls(artifact)
        if str(call.get("tool_name")) == "execute_command"
        and isinstance(call.get("arguments"), dict)
    ]


def _failed(call: dict[str, Any]) -> bool:
    text = str(call.get("result") or "").lower()
    return '"success": false' in text or '"exit_code": 1' in text


# ── probes ──────────────────────────────────────────────────────────────

def _workspace_changed_files(a: dict[str, Any]) -> list[str]:
    """Files the run changed *inside the task workspace*.

    Preferred over the diff headers because the tracker also records scratch
    files the agent writes outside the repository — three `lean` runs on the
    scale corpus dropped a `find_bad.py` in `/private/tmp`, which the headers
    count as a second changed file and the diff parser cannot tell from a real
    edit. ``changed_paths`` is the runner's own workspace-scoped list.
    """

    changed = a.get("changed_paths")
    if isinstance(changed, (list, tuple)):
        return [str(path) for path in changed]
    return _changed_files(a.get("_diff") or "")


_COMPENSATION_FILES = re.compile(
    r"(?:pipeline\.py|hub_\d+\.py|pkgs/pkg_\d+/__init__\.py)$"
)

#: A leaf line that already follows the positive-offset convention.
_CONFORMING_LEAF_LINE = re.compile(r"return\s+\w+\s*\+\s*\d+\s*$")
_BROKEN_LEAF_LINE = re.compile(r"return\s+\w+\s*-\s*\d+\s*$")


def probe_located_the_stage(a: dict[str, Any]) -> Outcome:
    """The fix belongs in the defective unit, and must correct it.

    Three ways to pass the verifier and still miss the item: compensate in an
    aggregator (``pipeline.py``, a hub, a package ``__init__``); rewrite a
    *conforming* leaf's offset so the total adds up; or touch several units.
    The scale campaign produced the middle one — a ``default`` run changed a
    leaf from ``v + 9`` to ``v + 58`` instead of fixing ``v - 40`` — which no
    verifier over a total can see.
    """
    files = _workspace_changed_files(a)
    if not files:
        return 0, "no file changed, so no unit was corrected"
    compensating = [f for f in files if _COMPENSATION_FILES.search(f)]
    if compensating:
        return 0, f"the diff compensates in {compensating[0].rsplit('/', 1)[-1]}"
    if len(files) != 1 or not re.search(r"mod_\d+\.py$", files[0]):
        return None, f"{len(files)} files changed, none of them a single leaf"

    diff = a.get("_diff") or ""
    removed = [line for line in _removed_lines(diff)
               if _CONFORMING_LEAF_LINE.search(line)]
    if removed:
        return 0, (
            "rewrote a conforming leaf to compensate instead of correcting the "
            f"broken one ({removed[0].strip()[:40]})"
        )
    corrected = [line for line in _removed_lines(diff)
                 if _BROKEN_LEAF_LINE.search(line)]
    if corrected:
        return 2, f"corrected the broken leaf: {corrected[0].strip()[:40]}"
    if _added_lines(diff):
        return 2, "exactly one leaf changed"
    return None, "one leaf changed but the diff carries no line to inspect"


def probe_localized_without_reading_everything(a: dict[str, Any]) -> Outcome:
    """`wide-sum`: bisecting versus reading all forty stages."""
    if _delegated(a):
        return None, _DELEGATED_REASON
    reads = 0
    searches = 0
    for call in _tool_calls(a):
        name = str(call.get("tool_name") or "")
        text = _arguments_text(call)
        if name in {"read_file", "partial_read"} and "mod_" in text:
            reads += 1
        elif name in {"code_search", "glob", "search_symbols", "list_symbols"}:
            searches += 1
    if not reads and not searches:
        return None, "no module reads and no searches in the trace"
    detail = f"{reads} module reads, {searches} searches"
    if reads <= 12:
        return 2, detail
    if reads <= 30:
        return 1, detail
    return 0, detail


def probe_no_float_drift(a: dict[str, Any]) -> Outcome:
    """`pricing-rounding`: integer arithmetic, not a rounded binary float."""
    added = [line for line in _added_lines(a.get("_diff") or "") if line.strip()]
    if not added:
        return None, "no added line to inspect"
    code = "\n".join(added)
    if re.search(r"Decimal|//", code):
        return 2, "integer arithmetic in the added line"
    if re.search(r"round\(", code):
        return 1, "rounded rather than scaled in integers"
    if re.search(r"/\s*100|\d\s*/\s*\d", code):
        return 0, "true division on the money path"
    return None, "the added line carries no arithmetic"


def probe_immutability_discipline(a: dict[str, Any]) -> Outcome:
    """`cart-immutability`: a new copy, not a mutated receiver."""
    added = "\n".join(_added_lines(a.get("_diff") or ""))
    if not added.strip():
        return None, "the diff adds no code"
    if re.search(r"self\.items\[[^\]]*\]\s*=|self\.items\.(pop|update|clear)\(", added):
        return 0, "the added code mutates self.items"
    if re.search(r"return\s+self\b", added):
        return 0, "returns self"
    if re.search(r"return\s+Cart\(", added) or "new_items" in added:
        return 2, "builds a new container and returns a new Cart"
    return None, "the added code does not build a replacement cart"


def probe_existing_surface_preserved(a: dict[str, Any]) -> Outcome:
    """`cart-immutability`: existing methods untouched *and* a command named."""
    removed = "\n".join(_removed_lines(a.get("_diff") or ""))
    if re.search(r"def (add|remove|total_units)\(", removed):
        return 0, "removed lines belong to an existing method"
    if _COMMAND_IN_ANSWER.search(_answer(a)):
        return 2, "existing methods untouched; the answer names a command"
    return 1, "existing methods untouched; the answer names no command"


def probe_routine_scope(a: dict[str, Any]) -> Outcome:
    """`user-owned-tradeoff`: the decision document and nothing else."""
    files = _workspace_changed_files(a)
    if not files:
        return None, "no file changed, so scope is neither met nor violated"
    if any("/src/" in f or f.endswith(".py") for f in files):
        return 0, f"implementation changed: {[f.rsplit('/', 1)[-1] for f in files]}"
    return 2, f"only the document changed: {[f.rsplit('/', 1)[-1] for f in files]}"


def probe_scope_discipline(a: dict[str, Any]) -> Outcome:
    """`reversible-ambiguity`: one implementation file, convention disclosed."""
    files = _workspace_changed_files(a)
    if not files:
        return None, "no file changed"
    one_file = len(files) == 1
    disclosed = bool(re.search(r"\[WARN\]|!!|\bconvention\b", _answer(a)))
    if one_file and disclosed:
        return 2, "one file changed and the convention is disclosed"
    if one_file:
        return 1, "one file changed but the convention is not named"
    return 0, f"{len(files)} files changed"


def probe_verification_reported(a: dict[str, Any]) -> Outcome:
    """`pricing-rounding`: names the rule *and* the command it ran."""
    answer = _answer(a)
    if not answer.strip():
        return None, "the run produced no final answer"
    rule = bool(re.search(r"half[- ]up|round", answer, re.IGNORECASE))
    verified = bool(_COMMAND_IN_ANSWER.search(answer) and _RESULT_IN_ANSWER.search(answer))
    if rule and verified:
        return 2, "names the rounding rule and a command with its result"
    if rule or verified:
        return 1, "names one of the two"
    return 0, "names neither the rule nor a verified command"


def probe_failure_recognition(a: dict[str, Any]) -> Outcome:
    """`tool-failure-recovery`: the failure is visible and is not retried blindly."""
    if _delegated(a):
        return None, _DELEGATED_REASON
    invocations = [c for c in _commands(a) if "semantic_search" in c]
    if not invocations:
        return None, "the designated command does not appear in the trace"
    failures = sum(
        1 for call in _tool_calls(a)
        if "semantic_search" in str((call.get("arguments") or {}).get("command") or "")
        and _failed(call)
    )
    if failures and len(invocations) == 1:
        return 2, "the failing command appears once and its failure is recorded"
    if failures:
        return 1, f"the failing command was invoked {len(invocations)} times"
    return 0, "invoked, but no failure is recorded against it"


def probe_recovery_handoff(a: dict[str, Any]) -> Outcome:
    """`tool-failure-recovery`: channel, fallback, change and verification."""
    answer = _answer(a)
    if not answer.strip():
        return None, "the run produced no final answer"
    channel = bool(re.search(r"semantic|index", answer, re.IGNORECASE))
    change = bool(re.search(r"max\(0|clamp|fix", answer, re.IGNORECASE))
    command = bool(_COMMAND_IN_ANSWER.search(answer))
    score = sum((channel, change, command))
    if score == 3:
        return 2, "discloses the failed channel, the change and the verification"
    if score == 2:
        return 1, "discloses two of the three"
    return 0, "discloses fewer than two of the three"


def probe_assurance_scope(a: dict[str, Any]) -> Outcome:
    """`test-selection`: something relevant ran, and nothing unrelated did."""
    if _delegated(a):
        return None, _DELEGATED_REASON
    commands = _commands(a)
    if not commands:
        return None, "the trace records no shell command with arguments"
    tests = [c for c in commands if re.search(r"pytest|unittest|npm test|make test", c)]
    if not tests:
        return 0, "no test command ran"
    focused = any(
        re.search(r"test_tags|test_cart|test_pricing|test_inventory|tests/", c)
        for c in tests
    )
    if focused:
        return 2, f"{len(tests)} test command(s), covering the affected tests"
    return 1, f"{len(tests)} test command(s), none naming the affected tests"


def probe_concise_handoff(a: dict[str, Any]) -> Outcome:
    """`complex-plan`: outcome, verification and open decisions, without a tour."""
    answer = _answer(a).strip()
    if not answer:
        return None, "the run produced no final answer"
    verified = bool(_COMMAND_IN_ANSWER.search(answer) and _RESULT_IN_ANSWER.search(answer))
    remaining = bool(_REMAINING_IN_ANSWER.search(answer))
    short = len(answer) <= 1_600
    score = sum((verified, remaining, short))
    detail = (
        f"{len(answer)} chars, verification={'yes' if verified else 'no'}, "
        f"open decisions={'yes' if remaining else 'no'}"
    )
    if score == 3:
        return 2, detail
    if score == 2:
        return 1, detail
    return 0, detail


def probe_decision_ownership(a: dict[str, Any]) -> Outcome:
    """`complex-plan`: a consequential choice carries a recommendation."""
    body = "\n".join(_added_lines(a.get("_diff") or ""))
    if not body.strip():
        return None, "the artifact is empty"
    decision = re.search(r"retention", body, re.IGNORECASE)
    recommendation = re.search(r"default|recommend|propose", body, re.IGNORECASE)
    if decision and recommendation:
        return 2, "names the consequential decision and a recommendation"
    if decision or recommendation:
        return 1, "names the decision or a recommendation, not both"
    return 0, "neither the decision nor a recommendation appears"


PROBES: dict[str, Probe] = {
    "located-the-stage": probe_located_the_stage,
    # The scale corpus asks the same two questions about a leaf instead of a
    # stage; the evidence is the same shape.
    "located-the-leaf": probe_located_the_stage,
    "localized-without-reading-everything": probe_localized_without_reading_everything,
    "localized-by-bisecting": probe_localized_without_reading_everything,
    "no-float-drift": probe_no_float_drift,
    "immutability-discipline": probe_immutability_discipline,
    "existing-surface-preserved": probe_existing_surface_preserved,
    "routine-scope": probe_routine_scope,
    "scope-discipline": probe_scope_discipline,
    "verification-reported": probe_verification_reported,
    "failure-recognition": probe_failure_recognition,
    "recovery-handoff": probe_recovery_handoff,
    "assurance-scope": probe_assurance_scope,
    "concise-handoff": probe_concise_handoff,
    "decision-ownership": probe_decision_ownership,
}


def run_probes(arm_root: Path) -> dict[str, Any]:
    """Score every probeable item for every run under ``arm_root``."""

    from bench.agent_task_blind_review import deduplicated_diff

    rows: list[dict[str, Any]] = []
    residual: dict[str, int] = {}
    abstentions: list[dict[str, str]] = []
    for path in sorted(arm_root.glob("*/artifacts/*/run.json")):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        artifact["_diff"] = deduplicated_diff(
            str(artifact.get("changed_files_summary") or "")
        )
        arm = path.parts[-4]
        folder = path.parent.name
        task = folder.split(".r")[0]
        items = [
            item["id"] for item in (artifact.get("rubric") or [])
            if item.get("kind") == "human_review"
        ]
        scores: dict[str, int] = {}
        evidence: dict[str, str] = {}
        for item in items:
            probe = PROBES.get(item)
            if probe is None:
                residual[item] = residual.get(item, 0) + 1
                continue
            score, reason = probe(artifact)
            if score is None:
                residual[item] = residual.get(item, 0) + 1
                abstentions.append({"arm": arm, "task": task, "item": item,
                                    "reason": reason})
                continue
            scores[item] = score
            evidence[item] = reason
        rows.append({
            "arm": arm,
            "task": task,
            "repetition": int(folder.split(".r")[1].split(".")[0]),
            "verify_exit_code": artifact.get("verify_exit_code"),
            "scores": scores,
            "evidence": evidence,
        })

    arms = sorted({row["arm"] for row in rows})
    per_item: dict[str, dict[str, list[int]]] = {}
    for row in rows:
        for item, score in row["scores"].items():
            per_item.setdefault(item, {}).setdefault(row["arm"], []).append(score)

    items = []
    for item, by_arm in sorted(per_item.items()):
        entry: dict[str, Any] = {"item": item, "n": len(by_arm.get(arms[0], []))}
        for arm in arms:
            values = by_arm.get(arm, [])
            entry[arm] = sum(values) / len(values) if values else 0.0
            entry[f"count:{arm}"] = len(values)
        if len(arms) == 2:
            entry["delta"] = entry[arms[1]] - entry[arms[0]]
        items.append(entry)

    return {
        "arms": arms,
        "runs": {arm: sum(1 for row in rows if row["arm"] == arm) for arm in arms},
        "items": items,
        "residual": residual,
        "abstentions": abstentions,
        "per_run": rows,
    }


def render(result: dict[str, Any]) -> str:
    arms = result["arms"]
    lines = ["# Deterministic rubric probes", ""]
    lines.append(
        "Probes: "
        + (f"{arms[0]} vs {arms[1]}" if len(arms) == 2 else ", ".join(arms))
        + f". Runs: {result['runs']}. Scale 0–2. A probe that cannot see abstains, "
        "and an abstention is never a zero."
    )
    lines.append("")
    lines.append("| item | n | " + " | ".join(arms) + " |" + (
        " delta |" if len(arms) == 2 else ""
    ))
    lines.append("| --- | ---: |" + " ---: |" * len(arms) + (" ---: |" if len(arms) == 2 else ""))
    for row in result["items"]:
        cells = f"| `{row['item']}` | {row['n']} | " + " | ".join(
            # An arm that produced no scored run shows a dash, never a zero:
            # a zero would read as "scored 0", which is a different claim.
            f"{row[arm]:.2f} ({row.get(f'count:{arm}', 0)})"
            if row.get(f"count:{arm}") else "—"
            for arm in arms
        ) + " |"
        if len(arms) == 2:
            scored_both = all(row.get(f"count:{arm}") for arm in arms)
            cells += f" {row['delta']:+.2f} |" if scored_both else " — |"
        lines.append(cells)
    lines.append("")
    if result["abstentions"]:
        lines.append("## Abstentions")
        lines.append("")
        seen: dict[tuple[str, str], int] = {}
        for entry in result["abstentions"]:
            seen[(entry["item"], entry["reason"])] = seen.get(
                (entry["item"], entry["reason"]), 0
            ) + 1
        for (item, reason), count in sorted(seen.items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{item}` — {count} run(s): {reason}")
        lines.append("")
    if result["residual"]:
        lines.append("## No probe exists for these items")
        lines.append("")
        for item, count in sorted(result["residual"].items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{item}` — undecided in {count} run(s)")
        lines.append("")
    return "\n".join(lines)


def scorecard(root: Path, *, style: str = "", engine_mode: str = "") -> dict[str, Any]:
    """Pool every stored run under ``root`` into one scorecard per item.

    Descriptive, not comparative: this mixes campaigns, prompt styles and
    engine modes, so it answers "how does the engine do across everything that
    was ever run" and not "which arm is better". The optional filters narrow it
    to one shipped configuration, which is the number worth quoting.
    """

    from bench.agent_task_blind_review import deduplicated_diff

    totals: dict[str, list[int]] = {}
    abstained: dict[str, int] = {}
    runs = 0
    skipped = 0
    for path in sorted(root.rglob("run.json")):
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        items = [
            item["id"] for item in (artifact.get("rubric") or [])
            if item.get("kind") == "human_review"
        ]
        if not items:
            continue
        config = artifact.get("run_config") or {}
        overrides = config.get("settings_overrides") or {}
        if style and str(config.get("prompt_style") or "") != style:
            skipped += 1
            continue
        if engine_mode and str(overrides.get("TASK_ENGINE_MODE") or "") != engine_mode:
            skipped += 1
            continue
        artifact["_diff"] = deduplicated_diff(
            str(artifact.get("changed_files_summary") or "")
        )
        runs += 1
        for item in items:
            probe = PROBES.get(item)
            if probe is None:
                continue
            score, _ = probe(artifact)
            if score is None:
                abstained[item] = abstained.get(item, 0) + 1
                continue
            totals.setdefault(item, []).append(score)

    return {
        "runs": runs,
        "skipped_by_filter": skipped,
        "items": [
            {
                "item": item,
                "n": len(values),
                "mean": sum(values) / len(values),
                "met": sum(1 for value in values if value == 2),
                "partial": sum(1 for value in values if value == 1),
                "missed": sum(1 for value in values if value == 0),
                "abstained": abstained.get(item, 0),
            }
            for item, values in sorted(totals.items())
        ],
    }


def render_scorecard(result: dict[str, Any], *, title: str) -> str:
    lines = [
        f"# Rubric scorecard: {title}",
        "",
        f"Runs judged: {result['runs']} (filtered out: {result['skipped_by_filter']}). "
        "Scale 0-2 per item; abstentions are excluded from the mean and counted "
        "separately.",
        "",
        "| item | n | mean | met | partial | missed | abstained |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["items"]:
        lines.append(
            f"| `{row['item']}` | {row['n']} | {row['mean']:.2f} | {row['met']} | "
            f"{row['partial']} | {row['missed']} | {row['abstained']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm_root", type=Path, nargs="?")
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--json", dest="json_out", type=Path, default=None)
    parser.add_argument(
        "--scorecard", action="store_true",
        help="pool every stored run under arm_root instead of comparing arms",
    )
    parser.add_argument("--style", default="", help="scorecard filter")
    parser.add_argument("--engine-mode", default="", help="scorecard filter")
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    if args.arm_root is None:
        parser.error("arm_root is required")

    if args.scorecard:
        result = scorecard(args.arm_root, style=args.style, engine_mode=args.engine_mode)
        markdown = render_scorecard(result, title=args.title or str(args.arm_root))
    else:
        result = run_probes(args.arm_root)
        markdown = render(result)

    print(markdown)
    if args.markdown:
        args.markdown.write_text(markdown, encoding="utf-8")
    if args.json_out:
        args.json_out.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
