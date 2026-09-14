"""Deterministic checks on the answer the user actually reads.

"Better communication with the user" was the one product goal with no number
behind it. Correctness is the verifier's job and tokens are counted from usage,
but the reply itself was only ever judged by a human reading it, which does not
scale to a campaign.

These checks are proxies, and they are worth stating as such: naming a command
is not the same as having run it, and a ``Verification:`` line is a format, not
a fact. What they measure is whether the answer is *checkable* — whether the
user can see what was run, what came back, and whether the answer stayed inside
the length the contract asks for. A reply that fails them is not necessarily
wrong; a reply that passes them is not necessarily right. They are cheap, they
are stable, and they move when the prompt contract changes, which is exactly
what an instrument for this goal has to do.

Usage::

    python -m bench.agent_task_answer_quality \
        bench/runs/RUN/arm-a/observations.jsonl \
        bench/runs/RUN/arm-b/observations.jsonl \
        --label-a baseline --label-b candidate
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

#: A command the answer says it ran, in or out of backticks.
_COMMAND = re.compile(
    r"`[^`]*(?:pytest|python|npm|cargo|go test|make|git|verify|node|ls|cat)[^`]*`"
    r"|(?:pytest|python -m|npm test|cargo test|go test|make test)",
    re.IGNORECASE,
)

#: An observed outcome, as opposed to a claim about one.
_RESULT = re.compile(
    r"\b(?:pass(?:ed|es)?|fail(?:ed|s)?|exit(?:ed)?[ _]?(?:code)?\s*\d|ok\b"
    r"|error|green\b)",
    re.IGNORECASE,
)

#: An answer that opens by narrating intent instead of stating the outcome.
_NARRATION = re.compile(
    r"^\s*(?:i will|i'll|let me|first,? i|i am going to|now i)", re.IGNORECASE
)

#: The length the loop protocol asks for when the deliverable is not a document.
_WORD_BUDGET = 250


def answer_quality(answer: str) -> dict[str, object]:
    """Score one final answer against the contract the prompt states."""
    text = answer or ""
    first = text.strip().split("\n", 1)[0][:160]
    return {
        "names_a_command": bool(_COMMAND.search(text)),
        "states_a_result": bool(_RESULT.search(text)),
        "has_verification_line": bool(
            re.search(r"verification\s*:", text, re.IGNORECASE)
        ),
        "opens_with_narration": bool(_NARRATION.match(first)),
        "over_word_budget": len(text.split()) > _WORD_BUDGET,
        "words": len(text.split()),
    }


def _answers(path: Path) -> dict[tuple[str, int], str]:
    """Read the final answer of every run in one observations file."""
    answers: dict[tuple[str, int], str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        artifact = Path(str(row.get("run_artifact", "")))
        text = ""
        if artifact.is_file():
            try:
                text = str(
                    json.loads(artifact.read_text(encoding="utf-8")).get(
                        "final_answer", ""
                    )
                    or ""
                )
            except (OSError, json.JSONDecodeError):
                text = ""
        answers[(str(row["task_id"]), int(row.get("repetition", 0)))] = text
    return answers


_BOOLEAN_CHECKS = (
    "names_a_command",
    "states_a_result",
    "has_verification_line",
    "opens_with_narration",
    "over_word_budget",
)


def compare(path_a: Path, label_a: str, path_b: Path, label_b: str) -> str:
    """Render the paired answer-quality table for two arms."""
    first, second = _answers(path_a), _answers(path_b)
    keys = sorted(set(first) & set(second))
    if not keys:
        raise ValueError("the two arms share no (task, repetition) pairs")

    lines = [
        f"# Answer quality: {label_a} vs {label_b}",
        "",
        f"Paired answers: {len(keys)}",
        "",
        f"| check | {label_a} | {label_b} |",
        "| --- | ---: | ---: |",
    ]
    for check in _BOOLEAN_CHECKS:
        count_a = sum(1 for key in keys if answer_quality(first[key])[check])
        count_b = sum(1 for key in keys if answer_quality(second[key])[check])
        lines.append(f"| {check} | {count_a}/{len(keys)} | {count_b}/{len(keys)} |")

    words_a = [int(answer_quality(first[key])["words"]) for key in keys]
    words_b = [int(answer_quality(second[key])["words"]) for key in keys]
    lines.append(
        f"| median words | {statistics.median(words_a):.0f} "
        f"| {statistics.median(words_b):.0f} |"
    )
    lines += [
        "",
        "These are proxies. Naming a command is not proof it ran, and a "
        "`Verification:` line is a format rather than a fact. What they show is "
        "whether the user can check the answer, which is what the goal can be "
        "held to without a human reading every reply.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm_a", type=Path)
    parser.add_argument("arm_b", type=Path)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()

    report = compare(args.arm_a, args.label_a, args.arm_b, args.label_b)
    print(report)
    if args.markdown:
        args.markdown.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
