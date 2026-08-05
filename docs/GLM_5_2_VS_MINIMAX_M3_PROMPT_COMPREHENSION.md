# GLM-5.2 versus MiniMax M3: prompt-comprehension comparison

## Bottom line

Neither model has one globally preferable “mind.” They optimize different interaction profiles.

- MiniMax is expansive, cautious in confidence, explicit about risks, and rich in stop conditions. It
  suits users who value visibility, control, and surfaced uncertainty, but it needs protection against
  semantic expansion and meta-analysis noise.
- GLM is concise, decisive, stable, and highly schema-reliable. It suits users who value speed and low
  interaction overhead, but it needs hard gates against confident invention and premature action.

The useful output is not a winner score. It is a set of prompt adaptations and routing choices for the
desired user outcome.

## Collection comparison

| Property | MiniMax M3 | GLM-5.2 |
|---|---:|---:|
| Planned calls | 672 | 672 |
| Structured successes | 668 | 671 |
| Parse failures | 4 | 1 |
| Provider/rate-limit failures | 0 | 0 |
| Mean latency | 16.01 s | 14.95 s |
| Mean confidence | 0.560 | 0.901 |
| Mean understanding length | about 54 words | about 42 words |
| Output tokens | 872,471 | 601,190 |

GLM produced about 31% fewer output tokens and better schema adherence. MiniMax’s additional tokens
mostly represent inferred constraints, ambiguities, verification steps, and risks rather than a
different central objective.

## Core differences

### Uncertainty

MiniMax’s confidence reacts strongly to vagueness, quantifiers, missing referents, and decision
support. GLM remains highly confident except under extreme vagueness. MiniMax’s confidence is more
informative as a diagnostic signal, but neither model consistently converts low confidence into
reduced authority.

### Expansion versus compression

MiniMax averages roughly four items in many fields whose authored key contains one. GLM usually
returns two or three. MiniMax is better at exposing possible failure modes and user decisions; GLM is
easier to consume and less likely to drown the harness in speculative policy.

### Ambiguity and autonomy

For the radically vague decision-support prompt, MiniMax authorizes clarification only. GLM
authorizes defining the outcome and inventing a model or UI. MiniMax is safer here.

For the ambiguous two-repository referent, both notice the ambiguity yet decide to inspect both. GLM
does so with materially higher confidence. Both require a hard referent gate.

### Indirect language

Both interpret “it would be useful to know how” as an operative request. GLM does so confidently;
MiniMax does so with reservations. Neither raw model reliably preserves the difference between
contextual interest, method explanation, drafting, and authorization to act.

### Conflict handling

Both detect equal-authority contradictions and impossible priority cycles. MiniMax explains the
authority reasoning at greater length and sometimes pulls the outer evaluation wrapper into the
conflict. GLM is cleaner but can place contradictory actions in `authorized_actions` while also
stopping execution. A downstream gate must read the whole structured state.

### Temporal authorization

MiniMax explicitly models future conditional permission. GLM emphasizes only what is authorized now.
MiniMax is more useful to a planner; GLM is conservatively useful to an immediate executor. The target
schema should encode both states rather than relying on either style.

### Epistemic integrity

Both follow explicit instructions to stop at the first research hint even after identifying the
conflict with reliable evidence. Both also accept retry-until-success and failure concealment. GLM is
more concerning in the latter case because it reports confidence 0.95 and does not flag a conflict;
MiniMax at least surfaces the transparency conflict. Neither should be trusted to infer an invariant
of truthful reporting without being told.

### Format robustness

GLM is more robust to nested instructions and schemas. MiniMax’s four failures cluster around a table,
a nested scope, an explicit exception, and a prompt requesting JSON output. This suggests instruction
collision between the embedded request and the outer response schema.

## Category routing implications

| Category | MiniMax tendency | GLM tendency | Suggested use |
|---|---|---|---|
| Planning | Broad discovery, many open decisions | Compact plan contract, confident target assumptions | MiniMax for collaborative planning; GLM after target is explicit |
| Implementation | Rich compatibility and test expansion | Concise executable interpretation | GLM for well-specified work; MiniMax when surfacing risks matters |
| Testing | Extensive evidence and blast-radius framing | Focused verification contract | Route by desired completeness versus speed |
| Code review | Detailed precedence and meta-awareness | Clean severity/evidence reconstruction | GLM for concise review; MiniMax for auditability |
| Web research | Rich ambiguity and source analysis | Stable, compact research contract | Both require an evidence-quality invariant |
| User interaction | More caveats and decision ownership | Direct conversion into interaction | MiniMax for control; GLM for low-friction UX after clear authorization |
| Decision support | Conservative under missing outcomes | Will define missing outcome despite low confidence | Prefer MiniMax until decisive criteria are supplied |
| External state | Explicit temporal state machine | Strong current-action gate | MiniMax for planning future steps; GLM for immediate safe execution |

## Implications for Infinidev prompt architecture

The results reinforce separating three responsibilities:

1. **Behavior prompt:** durable invariants such as truthful reporting, bounded recovery, authority
   discipline, and non-invention of user-owned outcomes.
2. **Execution-policy prompt:** planning cadence, evidence search, verification breadth, incremental
   work, and escalation rules.
3. **Objective prompt:** the actual task, deliverable, target, constraints, and user choices.

A single universal prompt should not try to erase model differences. Infinidev should select compact
model-specific adaptations:

- For MiniMax: constrain inference, separate harness metadata from task semantics, cap lists, and make
  uncertainty actionable.
- For GLM: make ambiguity and conflict hard execution gates, require explicit operative authority, and
  demand current/future authorization states.
- For both: enforce truthful failures, bounded retries, evidence adequacy, and no unsupported completion
  claims.

## Next analysis step

The reports characterize raw comprehension. They do not yet prove how each model executes a real
repository task. The next valid experiment is to choose a small set of model-specific behavior and
execution-policy candidates derived from these findings, then run held-out agent tasks. Optimize for a
declared user utility profile—control, speed, quality, or interaction level—rather than for a global
numeric winner.
