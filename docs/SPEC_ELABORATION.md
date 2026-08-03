# Specification Elaboration

Specification elaboration turns a non-trivial user request into a grounded
planning input before the planner creates implementation steps. It is an
optional enrichment phase: any failure returns the original escalation packet
and the pipeline continues.

## Runtime position

The orchestration pipeline runs elaboration after the chat agent escalates a
request and before council deliberation and planning. The resulting
`GroundedSpec` is attached to the `EscalationPacket` and rendered into the
planner handoff.

Requests shorter than `SPEC_ELABORATION_MIN_CHARS` skip the phase. Setting
`SPEC_ELABORATION_ENABLED=false` disables it entirely.

## Governing rule

Resolve technical uncertainty from repository or source evidence. Surface a
product choice only when multiple concrete implementations remain defensible
after that evidence is read. Never turn an answer already present in the task
or repository into a user question.

## Passes

The implementation in `engine/analysis/spec_elaborator.py` performs three model
passes followed by deterministic assembly:

1. Analyze the request into a deliverable, explicit scope, exclusions, and
   categorized gaps.
2. Ground gaps with a bounded set of read-only tools. The default budget is
   four calls. Unresolved gaps become explicit assumptions or product choices;
   they are never silently treated as facts.
3. Generate multiple design candidates. The default is three candidates.
4. Reject candidates whose existing-file or indexed-symbol claims fail
   deterministic checks, then assemble the surviving evidence and decision
   into a `GroundedSpec`.

All model passes use the single configured provider and model. The phase does
not require a second model or provider-specific behavior.

## Product-decision filter

A clarification reaches the user only when it includes:

- at least two concrete options;
- a default implementation that allows work to continue;
- a stated code impact for choosing another option.

`SPEC_ELABORATION_MAX_CLARIFICATIONS` limits how many decisions are surfaced in
one task. Additional candidates become explicit assumptions. A value of zero
turns every candidate into an assumption and prevents clarification output.

The planner must implement each stated default. It must not stall on the
question or silently select a third option.

## `GroundedSpec` contract

`engine/analysis/grounded_spec.py` owns the immutable handoff model. Its planner
render includes:

- deliverable, in-scope work, and exclusions;
- evidence-backed facts and unverified assumptions;
- product decisions with defaults;
- the selected design direction and rejected alternatives;
- risks and a retrieval signature.

`render_for_planner()` is the planner-facing contract. Keep it compact and
evidence-oriented; the planner needs the specification, not the elaboration
transcript.

## Configuration

All settings use the `INFINIDEV_` environment prefix:

| Setting | Default | Effect |
|---|---:|---|
| `SPEC_ELABORATION_ENABLED` | `true` | Enables the phase. |
| `SPEC_ELABORATION_MIN_CHARS` | `40` | Skips shorter requests. |
| `SPEC_ELABORATION_MAX_EVIDENCE_CALLS` | `4` | Caps read-only grounding calls. |
| `SPEC_ELABORATION_CANDIDATES` | `3` | Controls candidate design count, with a runtime minimum of two. |
| `SPEC_ELABORATION_MAX_CLARIFICATIONS` | `2` | Caps surfaced product decisions. |

## Failure behavior

The phase catches unexpected exceptions, clears its agent context, and returns
`None`. The pipeline then plans from the original escalation. This fallback is
load-bearing: elaboration must improve planning when it succeeds without
becoming a prerequisite for ordinary execution.

## Verification

Run the focused contract tests after changing this subsystem:

```bash
uv run pytest tests/test_spec_elaborator.py tests/test_pipeline_chat_to_planner.py -v
```

Also run `tests/test_prompt_style_rules.py` after changing any prompt or
planner-facing rendering in the phase.
