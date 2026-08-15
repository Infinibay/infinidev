"""Generate resumable silver task-policy proposals with a local causal LM.

The output intentionally uses a schema that is incompatible with human review
ledgers. Teacher proposals may prioritize annotation, but are never gold labels.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable


POLICIES = ("bugfix", "feature", "refactor", "performance", "research", "review")
UNCATEGORIZED_REASONS = (
    "acknowledgement",
    "status_only",
    "conceptual_question",
    "explanation_only",
    "quoted_action",
    "hypothetical_future",
    "meta_method",
    "ambiguous_authority",
    "out_of_domain",
    "unsupported_method",
    "ambiguous_method",
    "continuation_without_task",
    "conflicting_request",
    "insufficient_context",
    "reported_third_party_request",
    "healthy_existing_plan",
)
PROMPT_VERSION = "task-policy-silver-v1"
DEFAULT_MODEL = "Qwen/Qwen3-32B"
_JSON_OBJECT = re.compile(r"\{.*\}", flags=re.DOTALL)


ANNOTATION_GUIDELINES = """You annotate real user requests for conditional programming-agent methods.
Classify the requested outcome and authority, not keywords or the likely upstream patch.

Independent labels:
- bugfix: modify code to restore behavior that violates an existing contract.
- feature: modify code to create or change a capability, API, or observable behavior.
- refactor: modify internal structure while deliberately preserving observable behavior.
- performance: representative measurement is an independently requested method or outcome.
- research: external evidence, comparison, or an experiment is an independent outcome.
- review: evaluate a concrete artifact and report findings without modifying it.

Use multiple labels only when each independently changes the workflow. review is incompatible with
bugfix, feature, and refactor. Normal diagnosis before a fix is not research. Correct-but-slow is
performance; incorrect behavior is bugfix. Documentation, tests, migrations, dependency upgrades,
and configuration changes alone normally use no label because this taxonomy has no matching method.

If no label applies, choose exactly one uncategorized_reason from:
acknowledgement, status_only, conceptual_question, explanation_only, quoted_action,
hypothetical_future, meta_method, ambiguous_authority, out_of_domain, unsupported_method,
ambiguous_method, continuation_without_task, conflicting_request, insufficient_context,
reported_third_party_request, healthy_existing_plan."""

SYSTEM_PROMPT = ANNOTATION_GUIDELINES + """

Return one JSON object only, with keys policies, uncategorized_reason, confidence, rationale.
policies is an array of zero to three labels. uncategorized_reason is null when policies is nonempty.
confidence is a number from 0 to 1. rationale is one concise sentence about contract and authority."""


@dataclass(frozen=True)
class TeacherDecision:
    policies: tuple[str, ...]
    uncategorized_reason: str | None
    confidence: float
    rationale: str


def messages_for_request(issue_text: str) -> list[dict[str, str]]:
    """Build the teacher conversation without exposing upstream category hints."""
    if not issue_text.strip():
        raise ValueError("issue_text must not be empty")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"<user_request>\n{issue_text.strip()}\n</user_request>"},
    ]


def parse_teacher_decision(text: str) -> TeacherDecision:
    """Parse and strictly validate one teacher response."""
    match = _JSON_OBJECT.search(text)
    if match is None:
        raise ValueError("teacher response contains no JSON object")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError("teacher response contains invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("teacher response must be a JSON object")

    raw_policies = payload.get("policies")
    if not isinstance(raw_policies, list) or not all(
        isinstance(policy, str) for policy in raw_policies
    ):
        raise ValueError("policies must be a string array")
    policies = tuple(raw_policies)
    if len(policies) > 3 or len(set(policies)) != len(policies):
        raise ValueError("policies must contain at most three unique labels")
    unknown = set(policies) - set(POLICIES)
    if unknown:
        raise ValueError(f"unknown policies: {sorted(unknown)}")
    if "review" in policies and set(policies) & {"bugfix", "feature", "refactor"}:
        raise ValueError("review is incompatible with modifying policies")

    reason = payload.get("uncategorized_reason")
    if policies:
        if reason is not None:
            raise ValueError("labeled decisions must have null uncategorized_reason")
    elif reason not in UNCATEGORIZED_REASONS:
        raise ValueError("empty decisions require a known uncategorized_reason")

    confidence = payload.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be numeric")
    if not 0 <= float(confidence) <= 1:
        raise ValueError("confidence must be in [0, 1]")
    rationale = payload.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("rationale must be non-empty")
    return TeacherDecision(policies, reason, float(confidence), rationale.strip())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected object")
        rows.append(row)
    return rows


def _completed_ids(path: Path) -> frozenset[str]:
    if not path.exists():
        return frozenset()
    identifiers = set()
    for row in _read_jsonl(path):
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id:
            raise ValueError(f"{path}: proposal is missing candidate_id")
        if candidate_id in identifiers:
            raise ValueError(f"{path}: duplicate proposal for {candidate_id}")
        identifiers.add(candidate_id)
    return frozenset(identifiers)


def _batches(rows: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for index in range(0, len(rows), size):
        yield rows[index:index + size]


def generate_proposals(
    source: Path,
    output: Path,
    *,
    model_name: str,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    limit: int | None,
) -> dict[str, int]:
    """Generate append-only silver proposals, resuming an interrupted run."""
    if batch_size < 1 or max_input_tokens < 128 or max_new_tokens < 16:
        raise ValueError("invalid inference limits")
    rows = _read_jsonl(source)
    completed = _completed_ids(output)
    pending = []
    seen_source_ids = set()
    for row in rows:
        candidate_id = str(row.get("candidate_id", ""))
        issue_text = str(row.get("issue_text", ""))
        if not candidate_id or not issue_text.strip():
            raise ValueError("every source row needs candidate_id and issue_text")
        if candidate_id in seen_source_ids:
            raise ValueError(f"duplicate source candidate: {candidate_id}")
        seen_source_ids.add(candidate_id)
        if candidate_id not in completed:
            pending.append(row)
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        return {"source": len(rows), "already_completed": len(completed), "generated": 0}

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model.eval()

    output.parent.mkdir(parents=True, exist_ok=True)
    generated = 0
    parse_failures = 0
    with output.open("a", encoding="utf-8") as handle, torch.inference_mode():
        for batch in _batches(pending, batch_size):
            prompts = [
                tokenizer.apply_chat_template(
                    messages_for_request(str(row["issue_text"])),
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for row in batch
            ]
            inputs = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_input_tokens,
                return_tensors="pt",
            ).to(model.device)
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
            generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]
            responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            for row, response in zip(batch, responses, strict=True):
                proposal: dict[str, Any] = {
                    "candidate_id": row["candidate_id"],
                    "proposal_status": "silver_unreviewed",
                    "teacher_model": model_name,
                    "teacher_quantization": "bnb-nf4-double-quant",
                    "prompt_version": PROMPT_VERSION,
                }
                try:
                    decision = parse_teacher_decision(response)
                except ValueError as exc:
                    parse_failures += 1
                    proposal.update({
                        "parse_error": str(exc),
                        "raw_response": response.strip(),
                    })
                else:
                    proposal.update({
                        "proposed_policies": list(decision.policies),
                        "proposed_uncategorized_reason": decision.uncategorized_reason,
                        "confidence": decision.confidence,
                        "rationale": decision.rationale,
                    })
                handle.write(json.dumps(proposal, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                generated += 1
    return {
        "source": len(rows),
        "already_completed": len(completed),
        "generated": generated,
        "parse_failures": parse_failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report = generate_proposals(
        args.source,
        args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "ANNOTATION_GUIDELINES",
    "POLICIES",
    "PROMPT_VERSION",
    "TeacherDecision",
    "messages_for_request",
    "parse_teacher_decision",
]
