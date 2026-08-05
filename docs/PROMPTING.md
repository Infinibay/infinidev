# Prompting Policy

This document defines the writing standard for model-facing instructions in
Infinidev. The policy comes from the planner prompt, the strongest prompt in
the current runtime, and from the executable checks in
`tests/test_prompt_style_rules.py`.

The repository and the task packet contain facts that this policy cannot see.
When a general instruction conflicts with observed repository behavior, follow
the observed behavior and record the reason for the departure.

## Core standard

Write a prompt as a compact decision aid, not as a catalog of capabilities.
A reader must be able to distinguish facts and non-negotiable product bars
from methods that are useful defaults rather than universal obligations.

The governing rule is:

> Ground every action in the task packet, repository evidence observed this
> turn, or an earlier action whose output established the fact.

A remembered path, API, tool, or behavior is a hypothesis until current
evidence confirms it.

## Recommended structure

For procedural work, this is the default order:

1. State the input and its authority.
2. Turn unknowns into concrete observations.
3. Turn observations into actions.
4. Attach a verification check to every action that changes state.
5. Define completion and failure conditions.
6. End with one complete example when the format is not self-evident.

Depart from this order when another structure makes the task easier to verify;
record the reason in the prompt review.

## Rule design

Each rule must state the consequence of violating it. The consequence gives
the model a criterion it can apply to cases the prompt author did not predict.

Use a concrete anti-pattern and correction when wording alone leaves room for
interpretation:

- Bad: `Run the relevant tests.`
- Good: `Run the smallest test target that executes the changed behavior, then
  run the subsystem suite before reporting completion.`

Keep contracts and product bars imperative. Express methods as a default plus
the evidence that justifies departing from it. For example: `Prefer a focused
test first because it gives fast fault localization; run the broader suite
first when repository instructions make it the acceptance gate.`

## Machine facts, product bars, and methods

Label these three classes distinctly:

- Machine facts describe behavior the model cannot override, such as a tool
  ending the turn or a schema rejecting a call.
- Product bars remain mandatory under pressure: preserve user data, report
  actual verification results, and never claim unverified work is complete.
- Methods guide execution order and can be departed from when following them
  would make the result less verifiable. Record the observed reason whenever a
  method is departed from.

Do not present machine facts as advice or methods as immutable runtime facts.

## Forbidden wording

The prompt-style tests reject three classes of wording:

- Evasive uncertainty such as `perhaps`, `might`, `possibly`, and `try to`.
  It supplies neither a contract nor a usable recommendation. Words such as
  `prefer`, `generally`, and `usually` are valid for methods when the prompt
  also gives the decision criterion or departure condition.
- Threshold-free words such as `appropriate`, `relevant`, `as needed`,
  `reasonable`, `sufficient`, `proper`, and `significant`. Replace the word
  with the criterion it hides.
- Arrow glyphs such as `→`, `=>`, and `->` when prose can state the relation.

Quoted user questions, programming-language syntax, and machine-emitted text
may require narrow exemptions. Every exemption must identify the exact string
and explain why it is data or syntax rather than an instruction.

## Tool instructions

Derive tool names and parameters from the live public schemas. A prompt must
not name a retired tool, an internal compatibility alias, or a tool unavailable
to the role receiving the prompt.

For each tool call pattern, state:

- the question the call answers;
- the call or tool category;
- the evidence produced;
- the next decision that consumes that evidence.

This mapping prevents exploration that does not contribute to the result.

## Review checklist

Review every new or changed prompt in this order:

1. Name its governing rule in one sentence.
2. Confirm that section order matches execution order.
3. Confirm that every action traces to packet or repository evidence.
4. Confirm that every state-changing action has a verification condition.
5. Separate machine facts, product bars, and methods.
6. Replace evasive uncertainty and threshold-free words with explicit criteria;
   for a method, state its default and when evidence warrants departure.
7. Confirm all tool names and parameters against live schemas.
8. Include a complete example when the output format needs one.
9. Run `uv run pytest tests/test_prompt_style_rules.py`.
10. Run prompt integration tests for the affected runtime role.

The style test baseline is not an approval mechanism. Do not add new prompt
files to a debt or exemption list to make a failure disappear. Fix the prompt,
or document a narrow syntactic or quoted-data exemption with evidence.
