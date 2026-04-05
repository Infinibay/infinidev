"""Smart Context Summarizer - Generates compact summaries from action history."""

import re
from collections import defaultdict
from typing import List

from infinidev.engine.loop.models import ActionRecord, LoopState


CLASSIFIER_ID = "SMART_CONTEXT_SUMMARIZER"
MAX_SUMMARY_TOKENS = 200


class SmartContextSummarizer:
    """Generates condensed context summaries from agent action history.

    Analyzes action records to identify:
    - Modified files and their patterns
    - Error patterns and resolutions
    - Technical decisions made
    - Pending/unfinished work
    """

    def __init__(self, max_tokens: int = MAX_SUMMARY_TOKENS):
        self.max_tokens = max_tokens

    def generate_summary(self, state: LoopState) -> str:
        """Generate a compact summary of the current loop state.

        Args:
            state: Current loop state with history and plan

        Returns:
            Formatted summary string for system prompt injection
        """
        if not state.history or not state.plan:
            return ""

        summaries = self._analyze_actions(state.history)
        decisions = self._extract_decisions(state.history)
        errors = self._extract_errors(state.history)
        pending = self._extract_pending(state.plan, state.history)

        return self._format_summary(
            summaries=summaries,
            decisions=decisions,
            errors=errors,
            pending=pending,
            max_tokens=self.max_tokens,
        )

    def _analyze_actions(self, history: list[ActionRecord]) -> dict:
        """Group actions by file and detect patterns.
        
        Note: Uses summary text to infer file operations since ActionRecord
        doesn't expose detailed tool_call data.
        """
        files_modified = defaultdict(list)
        file_sequences = []
        current_sequence = []
        
        # Extract file paths from action summaries (format: "Read/write/edit <path>")
        file_patterns = [
            r"(?:Read|wrote|Wrote|edit|Edit|edited|EDITED)\s+([\w/._-]+)",
            r"(?:Read|wrote|Wrote|edit|Edit|edited|EDITED)\s+\`+([\w/._-]+)\`+",
        ]

        for record in history:
            # Try to extract file path from summary
            file_path = None
            for pattern in file_patterns:
                match = re.search(pattern, record.summary, re.IGNORECASE)
                if match:
                    file_path = match.group(1)
                    break

            if file_path:
                files_modified[file_path].append(record.summary)

                if current_sequence and current_sequence[-1] == file_path:
                    current_sequence.append(file_path)
                else:
                    if len(current_sequence) > 1:
                        file_sequences.append(current_sequence)
                    current_sequence = [file_path]

        if len(current_sequence) > 1:
            file_sequences.append(current_sequence)

        return {
            "files_modified": dict(files_modified),
            "frequently_edited": self._get_most_edited(files_modified, top=5),
            "consecutive_edits": file_sequences,
        }

    def _get_most_edited(self, files: dict, top: int = 5) -> List[tuple]:
        """Get most frequently edited files."""
        sorted_files = sorted(files.items(), key=lambda x: len(x[1]), reverse=True)
        return sorted_files[:top]

    def _extract_decisions(self, history: List[ActionRecord]) -> List[str]:
        """Extract technical decisions from action summaries."""
        decision_patterns = [
            r"(using|chose|decided to|switching to)\s+([\w_]+)",
            r"(replaced|changed|updated)\s+([\w_]+)\s+to\s+([\w_]+)",
            r"(added|removed)\s+(import|package|dependency)\s+([\w_]+)",
            r"(fixed|resolved|addressed)\s+(issue|bug|problem|error):?\s*([\w\s]+)",
        ]

        decisions = []
        for record in history:
            for pattern in decision_patterns:
                matches = re.findall(pattern, record.summary, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        decision = " ".join(filter(None, match))
                    else:
                        decision = match
                    if decision and len(decision) < 100:
                        decisions.append(decision)

        return list(set(decisions))[-10:]  # Keep last 10 unique decisions

    def _extract_errors(self, history: List[ActionRecord]) -> List[str]:
        """Extract error patterns and their resolutions."""
        error_keywords = ["error", "failed", "exception", "traceback", "not found"]
        resolution_keywords = ["fixed", "resolved", "solved", "handled", "workaround"]

        errors = []
        for i, record in enumerate(history):
            record_lower = record.summary.lower()

            has_error = any(kw in record_lower for kw in error_keywords)
            has_resolution = any(kw in record_lower for kw in resolution_keywords)

            if has_error and i > 0:
                prev_summary = history[i - 1].summary if i > 0 else ""
                if has_resolution or any(kw in prev_summary.lower() for kw in resolution_keywords):
                    error_summary = f"Resolved: {record.summary}"
                    if prev_summary:
                        error_summary += f" (after: {prev_summary[:50]})"
                    errors.append(error_summary)
            elif has_error:
                errors.append(f"Error detected: {record.summary}")

        return errors[-5:]  # Keep last 5 error patterns

    def _extract_pending(self, plan, history: List[ActionRecord]) -> List[str]:
        """Extract pending/unfinished work from plan."""
        pending_steps = []
        for step in plan.steps:
            if step.status in ["pending", "active"]:
                pending_steps.append(step.title)

        return pending_steps[:3]  # Keep first 3 pending items

    def _format_summary(
        self,
        summaries: dict,
        decisions: List[str],
        errors: List[str],
        pending: List[str],
        max_tokens: int = MAX_SUMMARY_TOKENS,
    ) -> str:
        """Format the final summary within token budget."""
        lines = []
        token_count = 0

        if summaries.get("frequently_edited"):
            lines.append("📝 MODIFIED FILES:")
            for file_path, count in summaries["frequently_edited"]:
                line = f"  • {file_path} ({count} edits)"
                if token_count + len(line) < max_tokens:
                    lines.append(line)
                    token_count += len(line)

        if decisions:
            lines.append("\n💡 TECHNICAL DECISIONS:")
            for decision in decisions:
                line = f"  → {decision}"
                if token_count + len(line) < max_tokens:
                    lines.append(line)
                    token_count += len(line)

        if errors:
            lines.append("\n⚠️ ERROR PATTERNS:")
            for error in errors:
                line = f"  ! {error}"
                if token_count + len(line) < max_tokens:
                    lines.append(line)
                    token_count += len(line)

        if pending:
            lines.append("\n📋 PENDING WORK:")
            for item in pending:
                line = f"  [ ] {item}"
                if token_count + len(line) < max_tokens:
                    lines.append(line)
                    token_count += len(line)

        return "\n".join(lines)[:max_tokens]
