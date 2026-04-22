"""Post-development code review engine.

Runs a single LLM call to review code changes after the developer loop
completes. If the review finds blocking issues, it provides feedback
that is fed back into the developer loop for fixes.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# Matches the per-file header emitted by engine.get_changed_files_summary()
# (see engine/loop/engine.py ~line 294): "### path (action)" optionally
# followed by ", no diff". This is the canonical source of truth for the
# file set — we grep it rather than asking the LLM to enumerate.
_FILE_HEADER_RE = re.compile(r"^###\s+(.+?)\s+\(([^)]+)\)\s*$", re.MULTILINE)


def _coerce_line(value: Any) -> int | None:
    """Coerce the judge's ``line`` field to an int or None.

    The LLM sometimes returns strings ("42"), sometimes floats, sometimes
    a range like "42-58". We accept the first integer we find; anything
    else becomes None.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    s = str(value).strip()
    if not s:
        return None
    m = re.search(r"-?\d+", s)
    if m is None:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


def _quote_is_grounded(
    quoted: str,
    file_path: str,
    file_contents: dict[str, str],
    file_changes_summary: str,
) -> bool:
    """Verify ``quoted`` appears verbatim in the cited file or the diff.

    Normalizes both sides by collapsing runs of whitespace so minor
    formatting drift doesn't trigger a false demotion. The string must
    still appear in the post-edit file contents (if we have them for
    ``file_path``) or anywhere in ``file_changes_summary``.

    Absent content and an absent quote both return False — the caller
    treats that as "couldn't verify" and demotes.
    """
    if not quoted:
        return False
    needle = _normalize_whitespace(quoted)
    if not needle:
        return False

    content = file_contents.get(file_path) if file_path else None
    if content and needle in _normalize_whitespace(content):
        return True
    if file_changes_summary and needle in _normalize_whitespace(file_changes_summary):
        return True
    return False


_WS_RE = re.compile(r"\s+")


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to single spaces for fuzzy containment."""
    return _WS_RE.sub(" ", text).strip()


def _compute_issue_id(issue: dict) -> str:
    """Assign a 10-char sha1 hash derived from the issue's grounded fields.

    Stable across the extractor's retry-once loop and across rework
    rounds — the same issue re-emitted gets the same id, enabling
    dedup and cross-round tracking.

    Uses the POST-validation severity so a demoted issue changes id;
    a demoted issue is effectively a different claim.
    """
    import hashlib
    desc = str(issue.get("description") or "")[:80]
    desc = _WS_RE.sub(" ", desc).strip().lower()
    parts = [
        str(issue.get("file") or "").strip(),
        str(issue.get("line") if issue.get("line") is not None else ""),
        str(issue.get("category") or "").strip().lower(),
        str(issue.get("severity") or "").strip().lower(),
        desc,
    ]
    key = "|".join(parts).encode("utf-8")
    return hashlib.sha1(key).hexdigest()[:10]


def _extract_changed_files_from_summary(summary: str) -> list[tuple[str, str]]:
    """Parse ``get_changed_files_summary`` output into (path, action) pairs.

    Deterministic ground truth for the diff's file set. Used to validate
    and backfill the extractor LLM's ``changes[]`` output, which is
    empirically prone to dropping files on large diffs.
    """
    if not summary:
        return []
    pairs: list[tuple[str, str]] = []
    for m in _FILE_HEADER_RE.finditer(summary):
        path = m.group(1).strip()
        action_raw = m.group(2).strip()
        # "modified, no diff" → action="modified"
        action = action_raw.split(",", 1)[0].strip()
        pairs.append((path, action))
    return pairs


# Multi-pass temperatures: extraction is maximally deterministic; the
# judge matches the long-standing single-pass value so verdicts stay
# consistent when we fall back.
EXTRACTOR_TEMPERATURE = 0.0
JUDGE_TEMPERATURE = 0.2


from infinidev.engine.analysis.review_result import ReviewResult

class ReviewEngine:
    """Post-development code review engine.

    Makes a single LLM call with the reviewer prompt to evaluate code
    changes. Returns APPROVED, REJECTED, or SKIPPED (if no code changes).
    """

    def __init__(self) -> None:
        self._review_count: int = 0
        self._max_reviews: int = 3  # Max review-rework cycles

    def reset(self) -> None:
        """Reset state for a new task."""
        self._review_count = 0

    def review(
        self,
        task_description: str,
        developer_result: str,
        file_changes_summary: str,
        *,
        file_reasons: dict[str, list[str]] | None = None,
        file_contents: dict[str, str] | None = None,
        recent_messages: list[str] | None = None,
        previous_feedback: str = "",
        plan_steps: list[dict] | None = None,
        automated_checks: dict[str, Any] | None = None,
        pre_extraction: dict | None = None,
    ) -> ReviewResult:
        """Review code changes and return a ReviewResult.

        Args:
            task_description: The original task given to the developer.
            developer_result: The developer's final answer.
            file_changes_summary: Unified diffs of all changed files.
            file_reasons: path → list of reasons for each changed file.
            file_contents: path → current content for each changed file.
            recent_messages: Last few conversation messages for context.
            previous_feedback: Feedback from a previous review round (for re-reviews).
            plan_steps: Ordered list of plan step dicts (title + explanation).
                When provided, the reviewer is asked to verify plan fidelity.
            automated_checks: Pre-computed deterministic check results keyed
                by check name. Shape: ``{"orphaned_references": [...],
                "missing_docstrings": [...], "verification_passed": bool}``.
                Items with severity='error' are BLOCKING by definition.
            pre_extraction: Pre-computed Pass A output. When provided, the
                reviewer skips its own extraction call and goes straight to
                judgment. Used by run_review_rework_loop to parallelize
                Pass A with ``collect_automated_checks``.

        Returns:
            ReviewResult with verdict and feedback.
        """
        from infinidev.flows.event_listeners import event_bus

        # Skip review if no code changes AND no files provided directly
        if not file_changes_summary.strip() and not file_contents:
            logger.info("ReviewEngine: no file changes and no files provided, skipping review")
            return ReviewResult(
                verdict="SKIPPED",
                summary="No file changes to review",
            )

        from infinidev.config.llm import get_litellm_params

        llm_params = get_litellm_params()
        if llm_params is None:
            logger.warning("ReviewEngine: no LLM params, skipping review")
            return ReviewResult(
                verdict="SKIPPED",
                summary="No LLM configured",
            )

        self._review_count += 1

        event_bus.emit("review_start", 0, "", {
            "round": self._review_count,
        })

        plan_steps = plan_steps or []
        automated_checks = automated_checks or {}
        file_reasons = file_reasons or {}
        file_contents = file_contents or {}
        recent_messages = recent_messages or []

        # Dispatch: multi-pass for complex diffs, single-pass otherwise.
        # ``pre_extraction`` lets run_review_rework_loop parallelize Pass A
        # with collect_automated_checks — when supplied, we skip the
        # extraction call and go straight to judgment.
        result: ReviewResult | None = None
        use_multi_pass = (
            pre_extraction is not None
            or self._should_multi_pass(file_changes_summary, file_contents)
        )
        if use_multi_pass:
            extraction = pre_extraction
            if extraction is None:
                extraction = self._run_extraction_pass(
                    task_description=task_description,
                    developer_result=developer_result,
                    file_changes_summary=file_changes_summary,
                    file_contents=file_contents,
                    plan_steps=plan_steps,
                )
            if extraction is not None:
                # Symbol grounding runs here (not inside extraction) because
                # `file_symbols` is part of automated_checks, which runs in
                # parallel with the extractor — they're both available now.
                self._ground_changes_against_symbols(
                    extraction, automated_checks.get("file_symbols") or {},
                )
                result = self._run_judgment_pass(
                    llm_params=llm_params,
                    extraction=extraction,
                    task_description=task_description,
                    developer_result=developer_result,
                    plan_steps=plan_steps,
                    automated_checks=automated_checks,
                    previous_feedback=previous_feedback,
                    recent_messages=recent_messages,
                )
            else:
                logger.info("ReviewEngine: extraction pass failed, falling back to single-pass")

        if result is None:
            result = self._single_pass_review(
                llm_params=llm_params,
                task_description=task_description,
                developer_result=developer_result,
                file_changes_summary=file_changes_summary,
                previous_feedback=previous_feedback,
                file_reasons=file_reasons,
                file_contents=file_contents,
                recent_messages=recent_messages,
                plan_steps=plan_steps,
                automated_checks=automated_checks,
            )

        # Ground every issue in the diff/file before anyone downstream
        # (feedback formatter, telemetry, future appeal) can trust them.
        self._validate_and_ground_issues(
            result,
            file_changes_summary=file_changes_summary,
            file_contents=file_contents,
        )

        event_bus.emit("review_complete", 0, "", {
            "verdict": result.verdict,
            "round": self._review_count,
            "issue_count": len(result.issues),
        })

        return result

    # ── Dispatch helpers ────────────────────────────────────────────────

    @staticmethod
    def _compute_complexity(
        file_changes_summary: str,
        file_contents: dict[str, str] | None,
    ) -> int:
        """Score = changed_lines + 50 * changed_files.

        `changed_lines` counts non-empty lines in the unified diff (a rough
        but robust proxy). `changed_files` uses the file_contents dict
        when available, else counts ``diff --git`` / ``--- a/`` headers.
        """
        lines = sum(
            1 for ln in file_changes_summary.splitlines()
            if ln.strip() and not ln.startswith("@@")
        )
        if file_contents:
            file_count = len(file_contents)
        else:
            # Fallback: count diff headers.
            file_count = file_changes_summary.count("\n--- a/") + (
                1 if file_changes_summary.startswith("--- a/") else 0
            )
            if file_count == 0:
                file_count = max(1, file_changes_summary.count("diff --git"))
        return lines + 50 * file_count

    def _should_multi_pass(
        self,
        file_changes_summary: str,
        file_contents: dict[str, str] | None,
    ) -> bool:
        """Decide single vs multi based on REVIEW_MULTI_PASS_MODE + threshold."""
        from infinidev.config.settings import settings as _settings
        mode = (getattr(_settings, "REVIEW_MULTI_PASS_MODE", "auto") or "auto").lower()
        if mode == "off":
            return False
        if mode == "always":
            return True
        threshold = int(getattr(_settings, "REVIEW_MULTI_PASS_COMPLEXITY_THRESHOLD", 400))
        return self._compute_complexity(file_changes_summary, file_contents) > threshold

    # ── Pass A: extraction ──────────────────────────────────────────────

    def _run_extraction_pass(
        self,
        *,
        task_description: str,
        developer_result: str,
        file_changes_summary: str,
        file_contents: dict[str, str],
        plan_steps: list[dict],
    ) -> dict | None:
        """Call Pass A LLM. Returns parsed JSON, or None on failure.

        One retry on malformed JSON with a 'return valid JSON' reminder.
        Any exception falls through as None so the caller can fall back to
        single-pass.
        """
        from infinidev.config.llm import get_litellm_params_for_review_extractor
        from infinidev.prompts.reviewer.extractor_system import EXTRACTOR_SYSTEM_PROMPT

        try:
            ext_params = get_litellm_params_for_review_extractor()
        except Exception as exc:
            logger.warning("ReviewEngine: extractor LLM params unavailable (%s)", exc)
            return None

        user_prompt = self._build_extractor_prompt(
            task_description=task_description,
            developer_result=developer_result,
            file_changes_summary=file_changes_summary,
            file_contents=file_contents,
            plan_steps=plan_steps,
        )
        messages = [
            {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        from infinidev.config.settings import settings as _s
        ext_provider = (_s.REVIEW_EXTRACTOR_LLM_PROVIDER or _s.LLM_PROVIDER or "")
        for attempt in range(2):
            try:
                response = self._completion_with_caching(
                    ext_params, messages, EXTRACTOR_TEMPERATURE, ext_provider,
                )
                raw = (response.choices[0].message.content or "").strip()
                parsed = self._parse_extraction(raw)
                if parsed is not None:
                    self._ground_changes_against_diff(parsed, file_changes_summary)
                    return parsed
            except Exception as exc:
                logger.warning("ReviewEngine: extractor LLM call failed: %s", exc)
                return None
            # Retry once with an explicit reminder.
            messages = messages + [
                {"role": "user", "content": "Your previous reply was not valid JSON. Return ONLY the JSON object specified in the schema."},
            ]

        return None

    @staticmethod
    def _ground_changes_against_diff(parsed: dict, file_changes_summary: str) -> None:
        """Ensure ``parsed['changes']`` covers every file in the diff.

        The extractor LLM sometimes omits files on large diffs (observed
        in practice: reports "3 test files" when 5 were added). We parse
        the canonical file set from the diff summary and append stub
        entries for anything the LLM missed, so the judge pass always
        sees the real file list — even if with minimal detail for the
        omitted ones. A warning is logged so the drop is observable in
        metrics without silently overriding a working extraction.

        Previously named ``_backfill_missing_files``; the new name
        reflects that this is one of a pair of grounding passes
        (:meth:`_ground_changes_against_symbols` handles symbol claims).
        """
        canonical = _extract_changed_files_from_summary(file_changes_summary)
        if not canonical:
            return

        changes = parsed.get("changes")
        if not isinstance(changes, list):
            changes = []
            parsed["changes"] = changes

        existing_paths = {
            str(c.get("file") or "").strip()
            for c in changes
            if isinstance(c, dict)
        }

        missing: list[tuple[str, str]] = [
            (path, action)
            for path, action in canonical
            if path and path not in existing_paths
        ]
        if not missing:
            return

        logger.warning(
            "Reviewer extraction dropped %d/%d files — backfilling: %s",
            len(missing), len(canonical),
            ", ".join(p for p, _ in missing[:5]) + ("…" if len(missing) > 5 else ""),
        )
        action_map = {"created": "added", "modified": "modified", "deleted": "deleted"}
        for path, action in missing:
            changes.append({
                "file": path,
                "kind": action_map.get(action, "modified"),
                "symbols_added": [],
                "symbols_removed": [],
                "line_range": "",
                "summary": "(present in diff but not captured by extractor; see diff for details)",
                "notable_lines": [],
                "_backfilled": True,
            })

    @staticmethod
    def _ground_changes_against_symbols(
        parsed: dict,
        file_symbols: dict[str, list[dict[str, Any]]],
    ) -> None:
        """Filter hallucinated entries out of ``symbols_added`` claims.

        The extractor LLM occasionally invents symbol names that aren't
        in the file — either misremembering the diff or confabulating
        "canonical" names. ``file_symbols`` is the post-edit ground
        truth from the code-intel index. For every change entry whose
        file has symbols indexed, we drop any ``symbols_added`` entry
        that isn't present in the real symbol list.

        ``symbols_removed`` is left untouched — we don't have
        pre-edit symbols in this phase (follow-up work), so we can't
        validate removal claims yet.

        A warning per filtered symbol keeps the drift observable.
        """
        if not file_symbols:
            return
        changes = parsed.get("changes")
        if not isinstance(changes, list):
            return

        for entry in changes:
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("file") or "").strip()
            if not path or path not in file_symbols:
                continue
            real_names = {s.get("name") for s in file_symbols[path] if s.get("name")}
            if not real_names:
                continue
            claimed = entry.get("symbols_added") or []
            if not isinstance(claimed, list):
                continue
            kept: list[str] = []
            dropped: list[str] = []
            for name in claimed:
                if isinstance(name, str) and name in real_names:
                    kept.append(name)
                else:
                    dropped.append(str(name))
            if dropped:
                logger.warning(
                    "Extractor symbols_added dropped %d hallucinated name(s) "
                    "for %s: %s (real symbols in file: %s)",
                    len(dropped), path,
                    ", ".join(dropped[:5]) + ("…" if len(dropped) > 5 else ""),
                    ", ".join(sorted(real_names)[:8]),
                )
                entry["symbols_added"] = kept
                entry.setdefault("_symbol_grounded", True)

    @staticmethod
    def _parse_extraction(raw: str) -> dict | None:
        """Parse the extractor's JSON output, tolerating markdown fences."""
        if not raw:
            return None
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            keep: list[str] = []
            in_block = False
            for ln in lines:
                if ln.strip().startswith("```") and not in_block:
                    in_block = True
                    continue
                if ln.strip() == "```" and in_block:
                    break
                if in_block:
                    keep.append(ln)
            text = "\n".join(keep) if keep else text
        from infinidev.engine.formats.tool_call_parser import safe_json_loads
        try:
            data = safe_json_loads(text)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        return data

    def _build_extractor_prompt(
        self,
        *,
        task_description: str,
        developer_result: str,
        file_changes_summary: str,
        file_contents: dict[str, str],
        plan_steps: list[dict],
    ) -> str:
        """Build the user prompt for the extractor pass."""
        parts: list[str] = []
        parts.append(f"## Original Task\n{task_description}")

        if plan_steps:
            plan_lines = ["## Plan"]
            for s in plan_steps:
                num = s.get("step", "?")
                title = s.get("title") or s.get("explanation", "")
                files = s.get("files") or []
                files_str = f" [{', '.join(files)}]" if files else ""
                plan_lines.append(f"{num}. {title}{files_str}")
                detail = s.get("explanation", "")
                if detail and detail != title:
                    plan_lines.append(f"   → {detail[:300]}")
            parts.append("\n".join(plan_lines))

        parts.append(f"## Developer's Report\n{developer_result}")

        if file_contents:
            parts.append("## Current File Contents")
            for path in sorted(file_contents.keys()):
                content = file_contents[path]
                max_chars = 50_000
                if len(content) > max_chars:
                    content = content[:max_chars] + f"\n... (truncated, {len(content)} total chars)"
                ext = path.rsplit(".", 1)[-1] if "." in path else ""
                parts.append(f"### `{path}`\n```{ext}\n{content}\n```")

        parts.append(f"## Diffs\n{file_changes_summary}")
        return "\n\n".join(parts)

    # ── Pass B: judgment ────────────────────────────────────────────────

    def _run_judgment_pass(
        self,
        *,
        llm_params: dict,
        extraction: dict,
        task_description: str,
        developer_result: str,
        plan_steps: list[dict],
        automated_checks: dict,
        previous_feedback: str,
        recent_messages: list[str],
    ) -> ReviewResult:
        """Call Pass B LLM with the extraction + checks (no diffs)."""
        from infinidev.prompts.reviewer.judge_system import JUDGE_SYSTEM_PROMPT

        user_prompt = self._build_judge_prompt(
            extraction=extraction,
            task_description=task_description,
            developer_result=developer_result,
            plan_steps=plan_steps,
            automated_checks=automated_checks,
            previous_feedback=previous_feedback,
            recent_messages=recent_messages,
        )
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            from infinidev.config.settings import settings as _s
            response = self._completion_with_caching(
                llm_params, messages, JUDGE_TEMPERATURE, _s.LLM_PROVIDER or "",
            )
            raw_content = response.choices[0].message.content or ""
            return self._parse_response(raw_content)
        except Exception as exc:
            logger.warning("ReviewEngine: judge LLM call failed (%s), skipping", exc)
            return ReviewResult(
                verdict="SKIPPED",
                summary=f"Judge skipped due to error: {exc}",
            )

    def _build_judge_prompt(
        self,
        *,
        extraction: dict,
        task_description: str,
        developer_result: str,
        plan_steps: list[dict],
        automated_checks: dict,
        previous_feedback: str,
        recent_messages: list[str],
    ) -> str:
        """Build the user prompt for the judge pass.

        Crucially, this does NOT include the raw diffs or file contents —
        the extraction is the authoritative summary the judge works from.
        """
        parts: list[str] = []

        if recent_messages:
            parts.append("## Conversation Context")
            for i, msg in enumerate(recent_messages):
                if i == len(recent_messages) - 1:
                    parts.append(f">>> CURRENT REQUEST <<<\n{msg}")
                else:
                    parts.append(msg)

        parts.append(f"## Original Task\n{task_description}")

        if plan_steps:
            plan_lines = ["## Plan (what the developer committed to)"]
            for s in plan_steps:
                num = s.get("step", "?")
                title = s.get("title") or s.get("explanation", "")
                files = s.get("files") or []
                files_str = f" [{', '.join(files)}]" if files else ""
                plan_lines.append(f"{num}. {title}{files_str}")
                detail = s.get("explanation", "")
                if detail and detail != title:
                    plan_lines.append(f"   → {detail[:300]}")
            parts.append("\n".join(plan_lines))

        if automated_checks:
            parts.append(self._format_automated_checks(automated_checks))

        parts.append("## Extraction (authoritative factual summary)")
        parts.append("```json\n" + json.dumps(extraction, indent=2) + "\n```")

        parts.append(f"## Developer's Report\n{developer_result}")

        if previous_feedback:
            parts.append(
                f"## Previous Review Feedback (Round {self._review_count})\n"
                f"The developer was asked to fix these issues. Verify they were addressed:\n"
                f"{previous_feedback}"
            )

        return "\n\n".join(parts)

    # ── Single-pass (legacy, also fallback) ─────────────────────────────

    def _single_pass_review(
        self,
        *,
        llm_params: dict,
        task_description: str,
        developer_result: str,
        file_changes_summary: str,
        previous_feedback: str,
        file_reasons: dict[str, list[str]],
        file_contents: dict[str, str],
        recent_messages: list[str],
        plan_steps: list[dict],
        automated_checks: dict,
    ) -> ReviewResult:
        """Classic one-shot reviewer. Also used as fallback from multi-pass."""
        from infinidev.prompts.reviewer.system import REVIEWER_SYSTEM_PROMPT

        user_prompt = self._build_review_prompt(
            task_description, developer_result,
            file_changes_summary, previous_feedback,
            file_reasons=file_reasons,
            file_contents=file_contents,
            recent_messages=recent_messages,
            plan_steps=plan_steps,
            automated_checks=automated_checks,
        )
        messages = [
            {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            from infinidev.config.settings import settings as _s
            response = self._completion_with_caching(
                llm_params, messages, JUDGE_TEMPERATURE, _s.LLM_PROVIDER or "",
            )
            raw_content = response.choices[0].message.content or ""
            return self._parse_response(raw_content)
        except Exception as exc:
            logger.warning("ReviewEngine: LLM call failed (%s), skipping review", exc)
            return ReviewResult(
                verdict="SKIPPED",
                summary=f"Review skipped due to error: {exc}",
            )

    def _build_review_prompt(
        self,
        task_description: str,
        developer_result: str,
        file_changes_summary: str,
        previous_feedback: str,
        *,
        file_reasons: dict[str, list[str]] | None = None,
        file_contents: dict[str, str] | None = None,
        recent_messages: list[str] | None = None,
        plan_steps: list[dict] | None = None,
        automated_checks: dict[str, Any] | None = None,
    ) -> str:
        """Build the user prompt for the review LLM call."""
        parts = []

        # Conversation context (last few messages, highlight the latest)
        if recent_messages:
            parts.append("## Conversation Context")
            for i, msg in enumerate(recent_messages):
                if i == len(recent_messages) - 1:
                    parts.append(f">>> CURRENT REQUEST <<<\n{msg}")
                else:
                    parts.append(msg)

        # Task context
        parts.append(f"## Original Task\n{task_description}")

        # Plan (what the developer committed to doing, step-by-step)
        if plan_steps:
            plan_lines = ["## Plan (what the developer committed to)"]
            for s in plan_steps:
                num = s.get("step", "?")
                title = s.get("title") or s.get("explanation", "")
                files = s.get("files") or []
                files_str = f" [{', '.join(files)}]" if files else ""
                plan_lines.append(f"{num}. {title}{files_str}")
                detail = s.get("explanation", "")
                if detail and detail != title:
                    plan_lines.append(f"   → {detail[:300]}")
            parts.append("\n".join(plan_lines))

        # Automated checks (deterministic evidence — pre-classified severity)
        if automated_checks:
            parts.append(self._format_automated_checks(automated_checks))

        # Developer's result
        parts.append(f"## Developer's Report\n{developer_result}")

        # Previous review feedback (for re-reviews)
        if previous_feedback:
            parts.append(
                f"## Previous Review Feedback (Round {self._review_count})\n"
                f"The developer was asked to fix these issues. Verify they were addressed:\n"
                f"{previous_feedback}"
            )

        # Files changed with reasons and current content
        if file_contents or file_reasons:
            parts.append("## Files Changed")
            all_paths = set(list((file_contents or {}).keys()) + list((file_reasons or {}).keys()))
            for path in sorted(all_paths):
                file_part = [f"### `{path}`"]
                reasons = (file_reasons or {}).get(path, [])
                if reasons:
                    file_part.append("**Reasons for changes:**")
                    for reason in reasons:
                        file_part.append(f"- {reason}")
                content = (file_contents or {}).get(path)
                if content is not None:
                    # Truncate very large files
                    max_chars = 50_000
                    if len(content) > max_chars:
                        content = content[:max_chars] + f"\n... (truncated, {len(content)} total chars)"
                    ext = path.rsplit(".", 1)[-1] if "." in path else ""
                    file_part.append(f"**Current content:**\n```{ext}\n{content}\n```")
                parts.append("\n".join(file_part))

        # The actual diffs to review
        parts.append(f"## Diffs\n{file_changes_summary}")

        return "\n\n".join(parts)

    def _parse_response(self, raw: str) -> ReviewResult:
        """Parse the LLM response into a ReviewResult."""
        raw = raw.strip()

        # Handle markdown code blocks
        if raw.startswith("```"):
            lines = raw.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```") and not in_block:
                    in_block = True
                    continue
                elif line.strip() == "```" and in_block:
                    break
                elif in_block:
                    json_lines.append(line)
            raw = "\n".join(json_lines)

        from infinidev.engine.formats.tool_call_parser import safe_json_loads
        try:
            data = safe_json_loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("ReviewEngine: could not parse response")
            return ReviewResult(
                verdict="SKIPPED",
                summary="Could not parse review response",
            )

        verdict = data.get("verdict", "APPROVED").upper()
        if verdict not in ("APPROVED", "REJECTED"):
            verdict = "APPROVED"

        return ReviewResult(
            verdict=verdict,
            summary=data.get("summary", ""),
            issues=data.get("issues", []),
            notes=data.get("notes", []),
        )

    @property
    def can_review_again(self) -> bool:
        """Whether we can do another review-rework cycle."""
        return self._review_count < self._max_reviews

    # ── Post-judge validation ───────────────────────────────────────────

    def _validate_and_ground_issues(
        self,
        result: ReviewResult,
        *,
        file_changes_summary: str,
        file_contents: dict[str, str],
    ) -> None:
        """Demote unsupported issues, verify citations, assign stable IDs.

        Applied to every ReviewResult right after the judge returns —
        covers both multi-pass and single-pass flows. Three checks per
        issue, in order:

        1. Blocking + non-structural without ``line`` or ``quoted_text``
           → demoted to ``important`` with a warning. The judge's
           instructions require citations; if it didn't follow, we don't
           trust it enough to reject on that issue alone.
        2. ``quoted_text`` present but not found in the diff or the
           post-edit file → demoted to ``suggestion`` + warning. A
           hallucinated quote is strong evidence the issue itself is a
           confabulation.
        3. ``id`` computed via deterministic hash (see
           ``_compute_issue_id``) so downstream code can dedup across
           retries without relying on LLM-stable ordering.
        """
        if not result.issues:
            return

        for issue in result.issues:
            if not isinstance(issue, dict):
                continue

            severity = str(issue.get("severity") or "").lower() or "important"
            category = str(issue.get("category") or "").lower() or "uncategorized"
            line = _coerce_line(issue.get("line"))
            quoted = str(issue.get("quoted_text") or "").strip()
            file_path = str(issue.get("file") or "").strip()

            issue["severity"] = severity
            issue["category"] = category
            issue["line"] = line
            issue["quoted_text"] = quoted

            # Rule 1: blocking non-structural issues must carry evidence.
            if severity == "blocking" and category != "structural":
                if line is None or not quoted:
                    logger.warning(
                        "Review issue demoted blocking→important (missing citation): "
                        "file=%s category=%s description=%.80s",
                        file_path, category, issue.get("description", ""),
                    )
                    issue["severity"] = "important"
                    issue.setdefault("_demoted", "missing_citation")

            # Rule 2: verify the quote actually appears in the diff or
            # post-edit file. Hallucinated quotes are strong evidence
            # the judge fabricated the issue itself.
            if quoted:
                found = _quote_is_grounded(
                    quoted, file_path, file_contents, file_changes_summary,
                )
                if not found:
                    logger.warning(
                        "Review issue demoted %s→suggestion (quoted_text not found in "
                        "file or diff): file=%s line=%s quote=%.60s",
                        issue["severity"], file_path, line, quoted,
                    )
                    issue["severity"] = "suggestion"
                    issue.setdefault("_demoted", "ungrounded_quote")

            # Rule 3: stable id. Computed from the FINAL severity so a
            # demotion changes the id — an issue's post-validation
            # identity is what downstream code should track.
            issue["id"] = _compute_issue_id(issue)

        # If every blocking issue got demoted, the verdict flag should
        # reflect reality. Leave the verdict string untouched (it's the
        # judge's authoritative output) but callers reading
        # ``is_rejected`` plus issue severities will now see consistency.
        blocking_left = [
            i for i in result.issues
            if isinstance(i, dict) and str(i.get("severity")).lower() == "blocking"
        ]
        if result.is_rejected and not blocking_left:
            logger.info(
                "All blocking issues were demoted during validation — verdict "
                "remains REJECTED but no blocking issues survive."
            )

    @staticmethod
    def _completion_with_caching(
        llm_params: dict,
        messages: list,
        temperature: float,
        provider_id: str,
        **extra,
    ):
        """Run litellm.completion with provider-aware prompt caching.

        The review prompts (extractor, judge, single-pass) are large and
        mostly static per session — they benefit massively from the same
        cache_control path the main loop uses.
        """
        import litellm
        from infinidev.config.prompt_cache import apply_prompt_caching

        call_kwargs = {
            **llm_params,
            "messages": messages,
            "temperature": temperature,
            **extra,
        }
        apply_prompt_caching(call_kwargs, provider_id)
        return litellm.completion(**call_kwargs)

    @staticmethod
    def _format_automated_checks(checks: dict[str, Any]) -> str:
        """Render automated check results as a compact, high-signal block.

        The reviewer is told to trust these — they came from deterministic
        tooling (tree-sitter index queries), not from another LLM.
        """
        lines = ["## Automated Checks (deterministic, trust these)"]

        verif = checks.get("verification_passed")
        if verif is True:
            lines.append("- tests/import-check: PASSED")
        elif verif is False:
            lines.append("- tests/import-check: FAILED (treat as blocking)")

        orphaned = checks.get("orphaned_references") or []
        lines.append(
            f"- orphaned_references: {len(orphaned)} "
            f"(BLOCKING — symbols removed but still referenced)"
        )
        for item in orphaned[:10]:
            lines.append(f"    • {item.get('message', '')}")
        if len(orphaned) > 10:
            lines.append(f"    • ... {len(orphaned) - 10} more")

        missing = checks.get("missing_docstrings") or []
        lines.append(
            f"- missing_docstrings: {len(missing)} "
            f"(suggestion — flag in notes, not sole reason to reject)"
        )
        for item in missing[:5]:
            lines.append(f"    • {item.get('message', '')}")
        if len(missing) > 5:
            lines.append(f"    • ... {len(missing) - 5} more")

        test_counts = checks.get("test_counts") or {}
        if test_counts:
            total_added = sum(
                max(0, v.get("delta", 0)) for v in test_counts.values()
            )
            lines.append(
                f"- test_counts: {len(test_counts)} test file(s), "
                f"{total_added} new test case(s) added in total"
            )
            for path, v in list(test_counts.items())[:15]:
                lines.append(
                    f"    • {path}: {v.get('before', 0)} → {v.get('after', 0)} "
                    f"(Δ {v.get('delta', 0):+d})"
                )

        file_symbols = checks.get("file_symbols") or {}
        if file_symbols:
            lines.append(
                f"- file_symbols: ground truth for extractor's symbols_added claims "
                f"({len(file_symbols)} file(s) covered)"
            )
            for path, syms in list(file_symbols.items())[:10]:
                names = ", ".join(s.get("name", "?") for s in syms[:8])
                more = f" (+{len(syms) - 8})" if len(syms) > 8 else ""
                lines.append(f"    • {path}: {names}{more}")

        hunk_stats = checks.get("hunk_stats") or {}
        if hunk_stats:
            total_add = sum(v.get("added", 0) for v in hunk_stats.values())
            total_rem = sum(v.get("removed", 0) for v in hunk_stats.values())
            lines.append(
                f"- hunk_stats: +{total_add} −{total_rem} across "
                f"{len(hunk_stats)} file(s)"
            )
            for path, v in list(hunk_stats.items())[:15]:
                lines.append(f"    • {path}: +{v.get('added', 0)} −{v.get('removed', 0)}")

        return "\n".join(lines)


def collect_automated_checks(
    changed_files: list[str],
    file_tracker: Any = None,
    verification_passed: bool | None = None,
    file_changes_summary: str = "",
) -> dict[str, Any]:
    """Gather deterministic check results to feed into the reviewer.

    Runs orphaned-reference and missing-docstring checks against the
    code-intel index for the given changed files. Resilient: any single
    check that errors returns an empty list for that key.

    Additional deterministic signals folded into the result:

    * ``test_counts`` — per test-file before/after/delta, computed by
      regex over the raw file content (see :mod:`.test_counter`). Grounds
      the judge against developer-report claims like "added 10 tests".
    * ``file_symbols`` — per changed file, the list of top-level symbols
      currently in the index. Catches extractor claims that a symbol was
      added/modified when no such symbol exists in the file.
    * ``hunk_stats`` — per file, the count of added/removed lines parsed
      from ``file_changes_summary``. A cheap sanity check on the
      extractor's ``line_range`` field.
    """
    from infinidev.code_intel.analyzer import (
        check_missing_docstrings, check_orphaned_references,
    )
    from infinidev.code_intel.smart_index import ensure_indexed

    project_id = 1  # CLI uses a single project by default
    result: dict[str, Any] = {
        "verification_passed": verification_passed,
        "orphaned_references": [],
        "missing_docstrings": [],
        "test_counts": {},
        "file_symbols": {},
        "hunk_stats": {},
    }

    # Reindex changed files so the checks see post-edit state
    import os as _os
    abs_changed = [_os.path.abspath(p) for p in (changed_files or [])]
    for p in abs_changed:
        try:
            ensure_indexed(project_id, p)
        except Exception as exc:
            logger.debug("Reindex failed for %s before checks: %s", p, exc)

    # Orphaned references — only if tracker recorded deletions
    if file_tracker is not None:
        try:
            deleted = file_tracker.get_deleted_symbols()
            if deleted:
                # Make sure source files are reindexed too
                for src in deleted.keys():
                    try:
                        ensure_indexed(project_id, src)
                    except Exception:
                        pass
                diags = check_orphaned_references(project_id, deleted)
                result["orphaned_references"] = [
                    {"file": d.file_path, "line": d.line, "message": d.message}
                    for d in diags
                ]
        except Exception as exc:
            logger.debug("Orphaned references collection failed: %s", exc)

    # Missing docstrings — one query per changed file
    for p in abs_changed:
        try:
            diags = check_missing_docstrings(project_id, p)
            for d in diags:
                result["missing_docstrings"].append(
                    {"file": d.file_path, "line": d.line, "message": d.message}
                )
        except Exception as exc:
            logger.debug("Missing-docstrings check failed for %s: %s", p, exc)

    # ── Deterministic signals for the judge ─────────────────────────────
    # These kill whole classes of LLM hallucination: "added 10 tests"
    # checked against regex count, "added function foo" checked against
    # the index, and "line_range 42-58" checked against actual +/- counts.
    try:
        result["test_counts"] = _compute_test_counts(abs_changed, file_tracker)
    except Exception as exc:
        logger.debug("test_counts check failed: %s", exc)

    try:
        result["file_symbols"] = _compute_file_symbols(project_id, abs_changed)
    except Exception as exc:
        logger.debug("file_symbols check failed: %s", exc)

    if file_changes_summary:
        try:
            result["hunk_stats"] = _compute_hunk_stats(file_changes_summary)
        except Exception as exc:
            logger.debug("hunk_stats check failed: %s", exc)

    return result


# ─────────────────────────────────────────────────────────────────────────
# Deterministic signal helpers
# ─────────────────────────────────────────────────────────────────────────


def _compute_test_counts(abs_changed: list[str], file_tracker: Any) -> dict[str, dict[str, int]]:
    """Count test cases before/after for each changed test file.

    Pulls before-content from the ``file_tracker``'s ``_originals`` and
    after-content from ``_current`` (both set as edits flow through
    ``FileChangeTracker.record``). Files not in the tracker fall back to
    reading the current file from disk, with ``before`` treated as empty.
    """
    from infinidev.engine.analysis.test_counter import (
        count_tests_for_files, looks_like_test_file,
    )

    entries: list[tuple[str, str | None, str | None]] = []
    for path in abs_changed:
        if not looks_like_test_file(path):
            continue
        before: str | None = None
        after: str | None = None
        if file_tracker is not None:
            try:
                # FileChangeTracker stores by abspath internally.
                before = file_tracker._originals.get(path)
                after = file_tracker._current.get(path)
            except Exception:
                pass
        if after is None:
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    after = f.read()
            except OSError:
                after = None
        entries.append((path, before, after))

    return count_tests_for_files(entries)


def _compute_file_symbols(project_id: int, abs_changed: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Return the top-level symbols currently indexed for each changed file.

    Uses the post-edit index (reindex runs before this helper). The judge
    uses this to validate extractor claims like ``symbols_added: ["foo"]``
    — if ``foo`` isn't in the file's symbol list, the judge knows the
    extractor hallucinated.

    Full before/after symbol delta is follow-up work; getting pre-edit
    symbols would require a separate tree-sitter pass on the tracker's
    ``_originals`` content and isn't worth the complexity for this phase.
    """
    from infinidev.code_intel.query import list_symbols

    out: dict[str, list[dict[str, Any]]] = {}
    for path in abs_changed:
        try:
            syms = list_symbols(project_id, path, limit=100)
        except Exception:
            continue
        if not syms:
            continue
        out[path] = [
            {"name": s.name, "kind": s.kind, "line": s.line_start}
            for s in syms
            # Filter to top-level-ish kinds; drop locals/params.
            if s.kind in ("class", "function", "method", "interface", "enum")
        ]
    return out


_HUNK_HEADER_RE = re.compile(r"^###\s+(.+?)\s+\(", re.MULTILINE)


def _compute_hunk_stats(file_changes_summary: str) -> dict[str, dict[str, int]]:
    """Count +/- lines per file by parsing the diff summary.

    The summary is the output of ``engine.get_changed_files_summary()``:
    sections headed by ``### path (action)`` followed by a fenced diff.
    We walk the string section by section, counting lines that start with
    ``+`` or ``-`` but aren't diff headers (``+++``/``---``).
    """
    stats: dict[str, dict[str, int]] = {}
    # Split on the file headers, keeping the path as the section key.
    positions = [(m.start(), m.group(1).strip()) for m in _HUNK_HEADER_RE.finditer(file_changes_summary)]
    if not positions:
        return stats
    positions.append((len(file_changes_summary), ""))  # sentinel end

    for i in range(len(positions) - 1):
        start, path = positions[i]
        end, _ = positions[i + 1]
        block = file_changes_summary[start:end]
        added = 0
        removed = 0
        for line in block.splitlines():
            if line.startswith("+++") or line.startswith("---"):
                continue
            if line.startswith("+"):
                added += 1
            elif line.startswith("-"):
                removed += 1
        stats[path] = {"added": added, "removed": removed}
    return stats


def run_review_rework_loop(
    *,
    engine: Any,
    agent: Any,
    session_id: str,
    task_prompt: tuple[str, str],
    initial_result: str,
    reviewer: ReviewEngine,
    recent_messages: list[str] | None = None,
    on_status: Any | None = None,
) -> tuple[str, ReviewResult | None]:
    """Run verification + review-rework cycle.

    1. Run tests/import checks (VerificationEngine)
    2. If tests fail, feed failure output back to developer for fixes
    3. Run textual code review (ReviewEngine)
    4. If rejected, feed feedback back to developer for fixes
    5. Loop until approved or max rounds reached

    Args:
        engine: LoopEngine (or any engine with ``execute()`` and file-change methods).
        agent: The developer agent.
        session_id: Current session ID for agent context activation.
        task_prompt: Original ``(description, expected_output)`` tuple.
        initial_result: The developer's first result string.
        reviewer: ReviewEngine instance (will be reset).
        recent_messages: Optional recent conversation summaries.
        on_status: Optional ``(level: str, message: str) -> None`` callback.
            Called with levels: "verification_pass", "verification_fail",
            "approved", "rejected", "max_reviews".

    Returns:
        ``(final_result, last_review)`` — the (possibly updated) result and
        the last ReviewResult, or None if review was skipped entirely.
    """
    def _notify(level: str, msg: str) -> None:
        if on_status:
            on_status(level, msg)

    def _run_verification_and_fix(current_result: str) -> str:
        """Run tests; if they fail, re-execute developer with failure context."""
        from infinidev.engine.analysis.verification_engine import VerificationEngine

        workspace = getattr(engine, '_workspace', None)
        if not workspace:
            from infinidev.tools.base.context import get_current_workspace_path
            workspace = get_current_workspace_path()

        if not workspace:
            return current_result

        verifier = VerificationEngine(workspace=workspace)
        changed = list((engine.get_file_contents() or {}).keys())
        tracker = getattr(engine, 'get_file_tracker', lambda: None)()
        vresult = verifier.verify(changed_files=changed, file_tracker=tracker)

        if vresult.passed:
            _notify("verification_pass", vresult.summary)
            return current_result

        # Tests failed — feed output back to developer
        _notify("verification_fail", vresult.summary)
        failure_feedback = vresult.format_for_developer()
        if not failure_feedback:
            return current_result

        fix_description = (
            f"{task_prompt[0]}\n\n"
            f"## IMPORTANT: Tests are FAILING\n"
            f"You MUST fix the test failures below before proceeding.\n\n"
            f"{failure_feedback}"
        )
        fix_prompt = (fix_description, task_prompt[1])

        agent.activate_context(session_id=session_id)
        try:
            new_result = engine.execute(
                agent=agent,
                task_prompt=fix_prompt,
                verbose=True,
            )
            return new_result if new_result and new_result.strip() else current_result
        finally:
            agent.deactivate()

    reviewer.reset()
    result = initial_result

    # Run verification before review (catch real breakage first)
    result = _run_verification_and_fix(result)

    previous_feedback = ""
    plan_steps = getattr(engine, "get_plan_steps", lambda: [])()

    while True:
        changed_files = list((engine.get_file_contents() or {}).keys())
        tracker = getattr(engine, "get_file_tracker", lambda: None)()
        file_changes_summary = engine.get_changed_files_summary()
        file_contents = engine.get_file_contents() or {}

        pre_extraction: dict | None = None
        # When multi-pass is in play, extraction (LLM-bound) and
        # collect_automated_checks (disk+SQLite-bound) are independent —
        # run both at once to save a couple of seconds per review round.
        if reviewer._should_multi_pass(file_changes_summary, file_contents):
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as ex:
                ext_future = ex.submit(
                    reviewer._run_extraction_pass,
                    task_description=task_prompt[0],
                    developer_result=result,
                    file_changes_summary=file_changes_summary,
                    file_contents=file_contents,
                    plan_steps=plan_steps,
                )
                checks_future = ex.submit(
                    collect_automated_checks,
                    changed_files=changed_files,
                    file_tracker=tracker,
                    verification_passed=True,
                    file_changes_summary=file_changes_summary,
                )
                pre_extraction = ext_future.result()
                automated = checks_future.result()
        else:
            automated = collect_automated_checks(
                changed_files=changed_files,
                file_tracker=tracker,
                verification_passed=True,
                file_changes_summary=file_changes_summary,
            )

        review = reviewer.review(
            task_description=task_prompt[0],
            developer_result=result,
            file_changes_summary=file_changes_summary,
            file_reasons=engine.get_file_change_reasons(),
            file_contents=file_contents,
            recent_messages=recent_messages or [],
            previous_feedback=previous_feedback,
            plan_steps=plan_steps,
            automated_checks=automated,
            pre_extraction=pre_extraction,
        )

        if review.is_approved:
            _notify("approved", review.summary)
            return result, review

        if not review.is_rejected:
            # SKIPPED
            return result, review

        # Rejected
        feedback = review.format_feedback_for_developer()
        _notify("rejected", review.format_for_user())

        if not reviewer.can_review_again or not feedback:
            _notify("max_reviews", "")
            return result, review

        # Re-execute the developer with feedback prepended to the prompt
        fix_description = (
            f"{task_prompt[0]}\n\n"
            f"## IMPORTANT: Code Review Feedback\n"
            f"Your previous implementation was REJECTED by the reviewer. "
            f"You MUST address ALL blocking issues below before proceeding.\n\n"
            f"{feedback}"
        )
        fix_prompt = (fix_description, task_prompt[1])
        previous_feedback = feedback

        agent.activate_context(session_id=session_id)
        try:
            result = engine.execute(
                agent=agent,
                task_prompt=fix_prompt,
                verbose=True,
            )
            if not result or not result.strip():
                result = "Done. (no additional output)"
        finally:
            agent.deactivate()
