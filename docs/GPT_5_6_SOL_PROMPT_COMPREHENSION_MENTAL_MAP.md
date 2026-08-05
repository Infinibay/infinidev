# GPT-5.6 Sol prompt-comprehension mental map

## Scope and evidence

This report describes GPT-5.6 Sol's externally observable prompt reconstructions. It does not claim
access to hidden chain-of-thought. The campaign contained 672 isolated calls: every case used a fresh
conversation, no system prompt, and one user message containing the request under test plus a fixed
instruction to avoid execution and return a structured understanding.

- Dataset: 672 cases, 224 controlled three-variant families, eight domains.
- Structured successes: 672/672.
- Parse, provider, and rate-limit failures: 0.
- Mean/median latency: 20.67 / 19.63 seconds.
- Input/output tokens reported: 186,890 / 594,563.
- Mean self-reported confidence: 0.9744.
- Dataset SHA-256: `161bee159a087f349c513ddfb18fff9f7074d7962193ca918be51e09e20ac0d1`.
- Ledger SHA-256: `4bd51dbfe6ce92b06179fa1ba1ce4c1cc23fa827cca45bf5d7dc6a5ad26fc071`.
- Derived evidence-report SHA-256:
  `5b10967c44dc2165c48beacaaaaf171a7220bd4f564fdba6522443762ef7cdf6`.

The lossless evidence is in
`bench/runs/gpt-5.6-sol-prompt-comprehension-full/evidence-report.{json,md}`. Numeric summaries below
describe response tendencies, not a global quality score.

## Executive mental model

Sol behaves like a highly confident policy-and-requirements interpreter. Its free reconstruction is
compact, but its structured result is expansive: it translates a short request into a detailed set of
constraints, authorized and unauthorized actions, verification requirements, ambiguities, stop
conditions, and interpretation risks.

It is especially strong at explicit authority boundaries, same-level contradictions, future
conditional permission, evidence-limited completion claims, and separating local work from external
state changes. It preserves the central task reliably across formality, formatting, instruction
position, typos, and code switching.

Its principal weaknesses are:

1. Confidence is almost saturated, including for underspecified and contradictory prompts.
2. It often treats the outer comprehension instruction as part of the embedded task's authority
   hierarchy, contaminating the requested “raw” reconstruction.
3. It converts indirect interest into an operative request.
4. It detects ambiguous referents but may broaden the target rather than stop immediately.
5. It expands one stated rule into many inferred policies that may be useful but are not literal prompt
   meaning.

The characteristic process visible in its answers is:

1. Identify the embedded objective and artifact.
2. Apply the outer non-execution instruction as the controlling instruction for the current response.
3. Construct a future-execution contract for what would be allowed later.
4. Enumerate evidence, safety boundaries, ambiguities, and stop conditions.
5. Resolve authority explicitly and report near-certain confidence.

## Structural tendencies

The authored interpretation keys contain approximately one item per structured field. Sol returns an
average of 2.57 deliverables, 4.17 constraints, 2.21 user-owned decisions, 3.62 authorized actions,
3.99 unauthorized actions, 4.23 verification requirements, 3.28 ambiguities, 2.98 stop conditions,
0.54 conflicts, and 3.76 interpretation risks.

This is not merely verbosity: Sol creates an execution-ready policy model around the task. That can be
valuable inside Infinidev, but the added material must be labeled as model-derived defaults rather than
user-supplied requirements.

Confidence is not discriminative enough for execution gating. Of 672 answers, 670 report confidence at
or above 0.8. Even vagueness—the lowest-confidence phenomenon—averages 0.936. A confidence value from
Sol should be interpreted mainly as confidence in its reconstruction, not proof that the request is
complete, executable, or safely authorized.

## Methodological finding: wrapper assimilation

Sol frequently incorporates the fixed evaluation wrapper into the embedded task. It adds items such as
“for this response, report only an understanding,” marks the embedded execution request as conflicting
with the surrounding non-execution instruction, and gives the wrapper priority.

This behavior is logically defensible—the wrapper is part of the only user message—but it means the
campaign does not expose a perfectly unconditioned mental model. The responses reveal how Sol composes
nested instructions, not only how it understands the request-under-test in isolation.

This effect is stronger and more systematic than in GLM, and more consistently structured than in
MiniMax. A future instrument should separate semantic elicitation from the stimulus through an API
mechanism such as constrained output or an evaluator turn, or explicitly ask the model to reconstruct
only the delimited request without adding the elicitation wrapper to its semantic fields.

## Category analysis

### Planning

Sol understands planning as read-only discovery followed by a repository-grounded plan. It reliably
preserves touchpoints, tests, rollout, rollback, open API decisions, and the prohibition on source
changes. It also stops before implementation and treats implementation as separately authorized work.

The important failure is target resolution. In
`missing_context_and_referents--planning--v1--ambiguous-workspace-target`, two repositories are named
but “this repository” has no unique referent. Sol identifies the ambiguity yet authorizes future
read-only inspection of both repositories and reports confidence 0.98. It does later require
clarification if evidence cannot establish direction, but it has already broadened the singular target.
The behavior prompt should prohibit replacing an unresolved target with the union of all candidates.

### Implementation

Sol consistently retains compatibility, legacy-path preservation, local verification, and deployment
boundaries. It correctly distinguishes a format example from normative behavior: the requested
Summary/Evidence headings survive, while deployment shown in the example does not become authorized.

For retry-and-hide failure recovery, Sol follows the instruction to omit transient failures and retry
until success. Unlike a naive interpretation, it adds that completion cannot be claimed if the lookup
never succeeds and that persistent or unsafe failure requires a pause. This is safer than GLM's raw
response, but it still accepts omission of material attempt history. Truthful reporting and bounded
retry should remain explicit behavior invariants.

Implementation has Sol's lowest category confidence, but it is still 0.960.

### Testing and verification

Sol gives this domain its longest free reconstructions, averaging 48.8 words. It reliably distinguishes
test selection, executed evidence, what results establish, impacted coverage, and whether a contract
change justifies editing tests.

For `some explicitly named requirements` when none are named, Sol identifies the missing requirements,
refuses unsupported verification claims, and stops for clarification. It expands the test policy with
broader regression checks and evidence-linked completion conditions. These are useful execution
defaults but exceed literal reconstruction.

### Code review

Sol strongly separates review from remediation. It preserves severity ordering, exact evidence,
impact, and no-fix boundaries. It detects a same-level fix/no-fix contradiction, refuses to choose a
winner, and requires user clarification before modification.

Its structured handling is internally coherent: future review can be authorized while current
execution is prohibited by the wrapper, and code fixes remain conditional on resolving the embedded
conflict. The drawback is wrapper assimilation, which adds a meta-conflict that would not exist in a
real review request.

### Web research

Sol preserves primary sources, currentness, compatibility, citations, and the no-dependency-change
boundary. It separates research and recommendation from implementation.

In the adversarial first-hint case, Sol detects that accepting an unchecked hint conflicts with a
current primary-source-supported conclusion. It gives the local first-hint rule procedural force but
does not allow it to justify unsupported claims: if the hint is inadequate, it pauses or hands off with
qualification. This is a stronger epistemic default than the raw GLM and MiniMax responses.

### User interaction

Sol reliably keeps decisive cost-versus-control priorities with the user and stops after presenting the
options and asking the focused question. It does not infer permission to implement the resulting
choice.

However, it treats “it would be useful to know how” and “might help” as authorization to explain and
formulate the decision question, with confidence 0.96. It notes that the actual approaches are missing,
but the weak implication still becomes an operative request. The behavior layer must distinguish
context, desire for method knowledge, drafting, and authority to interact or act.

### Decision support

Sol is conservative about user-owned outcomes. For “make the decision support situation better; use
your judgment,” it restricts current authority to interpretation and discovery, prohibits assuming
unstated priorities, and requires clarification before implementation. This is safer than GLM's broad
authorization to define “better.”

For mutually exclusive equal priorities, it identifies the deadlock and refuses to invent a tie-breaker.
It requires a revised sequencing rule, permission to balance objectives, or a direct user decision.

### Authorization and external state

Sol represents authorization as a temporal state machine. Local release preparation and verification
are authorized; publication, upload, tags, and messages remain unauthorized until explicit approval;
after approval, only the specifically approved external actions become authorized. Completion of local
verification is explicitly not treated as implicit approval.

This is the most complete representation among the three tested models for a planner that must reason
about current and future authority. Sol also adds strong handoff and credential boundaries.

## Linguistic findings

- Register, structure, and instruction position have little effect on the core objective and boundary.
- Uppercase emphasis does not materially strengthen an already hard prohibition.
- Real exceptions and temporal triggers are represented separately from current permission.
- Negation scope and same-level contradictions are handled reliably.
- Examples are treated as subordinate to explicit normative rules.
- Irrelevant anecdotes are generally excluded from the task contract.
- Typos and Spanish-English code switching preserve central meaning.
- Quantifier changes affect verification scope, but confidence remains almost unchanged.
- Radical vagueness creates clarification and stop conditions but only a small confidence reduction.
- Weak implication is over-promoted to action authority.
- Missing referents are surfaced but may be broadened into multi-target inspection.
- Output-format compliance is perfect in this campaign, including embedded JSON and table cases.

## Execution-policy findings

Sol naturally supplies discovery, planning, incremental work, verification, safe local preparation,
rollback awareness, and escalation. It refuses unsupported completion claims and handles impossible
priority conflicts well. Its first-hint response shows a meaningful epistemic backstop: a procedural
shortcut does not authorize an unsupported conclusion.

It still requires explicit invariants for truthful reporting and bounded recovery. It can follow a
request to hide transient failures, and its expansive inferred policy can obscure which requirements
came from the user versus from the model.

## Recommended prompt adaptations

1. Tell Sol to separate literal user requirements from model-derived defaults and label both.
2. Prohibit broadening an unresolved singular target to every plausible candidate.
3. Require an operative request before converting indirect interest into authorization.
4. Encode truthful reporting of material failures and bounded retry as behavior invariants.
5. Preserve Sol's useful evidence rule: a local shortcut cannot justify an unsupported claim.
6. Keep explicit current, future-conditional, and trigger-satisfied authorization states.
7. Treat ambiguity, conflicts, and stop conditions as execution gates; ignore saturated confidence as a
   safety signal.
8. Prevent the harness's elicitation instructions from being reproduced as task conflicts or task
   constraints.
9. Keep objective prompts concise. Sol already expands them into a detailed execution contract; more
   generic prose is likely to increase redundant policy rather than comprehension.
10. For user-control profiles, retain explicit handoffs. For speed profiles, constrain the number of
    inferred ambiguities and verification items without weakening hard boundaries.

## Comparison-oriented conclusion

Sol combines GLM's decisiveness with much of MiniMax's policy expansion. It is more schema-reliable than
both in this run and has the strongest explicit temporal authorization model. It is also the most
confident model and the most systematic at assimilating the evaluation wrapper.

For Infinidev, Sol is a strong default for complex agent work once the prompt makes three things
unambiguous: the exact target, what constitutes operative authorization, and which behavioral
invariants cannot be overridden by task-local instructions. Its confidence should never substitute for
those gates.

## Reliability boundary

All 672 calls produced structurally valid records, so every family has complete evidence. The main
limitation is not missing data but measurement contamination from the comprehension wrapper. The
report describes observable reconstruction behavior under that wrapper and should not be presented as
hidden reasoning or as an unconditional model personality.
