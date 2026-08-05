# Evidence map for the model-behavior laboratory

This document separates three different things that should never be collapsed:

- a paper's empirical result in its own setting;
- an Infinidev design inference bounded by that result;
- an unverified product hypothesis.

The machine-readable source of truth is `bench/model_behavior_evidence.json`. Run
`uv run python -m bench.evidence_registry bench/model_behavior_evidence.json --root .` to reject
unknown citations, missing internal artifacts, unsupported “supported” claims, or decisions without
scope and limitation statements.

## Primary research used

- [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) supports standardized,
  multi-scenario, multi-metric evaluation and publication of raw prompts/completions. It does not
  validate this project's taxonomy.
- [Large Language Models Are Not Robust Multiple Choice Selectors](https://arxiv.org/abs/2309.03882)
  demonstrates option-ID/position sensitivity across 20 LLMs and motivates complete action-position
  counterbalancing. It does not guarantee that cyclic rotation removes every bias.
- [Fantastically Ordered Prompts and Where to Find Them](https://arxiv.org/abs/2104.08786) shows that
  few-shot example order can radically alter performance and that good orders may not transfer
  between models. It supports controlling order effects, not a universal claim that isolated calls
  are always the best deployment evaluation.
- [PromptRobust](https://arxiv.org/abs/2306.04528) demonstrates failures under small
  semantics-preserving prompt perturbations. It motivates paired variants, while independent review
  must still verify that our variants change only the intended factor.
- [Language Models Don't Always Say What They Think](https://arxiv.org/abs/2305.04388) shows that
  chain-of-thought explanations can omit influential biasing features and rationalize outcomes. It
  motivates separating self-report from choice-only behavior; it does not make every user-facing
  criterion worthless.
- [Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409) and
  [DSPy](https://arxiv.org/abs/2310.03714) show that automatic prompt search or compilation can
  outperform hand-written baselines on tested tasks. Neither establishes universal transfer or
  safe activation without held-out regression gates.
- [TextGrad](https://arxiv.org/abs/2406.07496) demonstrates textual feedback as an optimization
  signal in compound LM systems. It motivates turning critiques into candidate experiments, not
  trusting model self-critique or applying its proposed changes directly.
- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
  evaluates held-out human preference and public capability/safety regressions, and explicitly notes
  that labeler interpretations can diverge from the actual user's intent. It supports keeping
  user-specific objectives visible, not equating runtime prompting with RLHF.
- [Lost in the Middle](https://arxiv.org/abs/2307.03172) shows that relevant-information position
  can materially affect long-context performance. It motivates testing ranked, bounded context
  against full-context placements; it does not validate ContextRank or its current ceiling.
- [Guidelines for Human-AI Interaction](https://doi.org/10.1145/3290605.3300233) validated the
  relevance of 18 human-AI design guidelines through multiple rounds, including 49 practitioners
  evaluating 20 AI-infused products. This supports making actual runtime state visible, but not
  Infinidev's exact wording or cadence.
- [The impact of progress indicators on task completion](https://doi.org/10.1016/j.intcom.2010.03.001)
  found that feedback effects depend on whether displayed progress matches expectations and that
  intermittent feedback can avoid some harms. Because coding-agent work has no reliable linear
  completion percentage, Infinidev reports real phase transitions and elapsed time rather than a
  fabricated percentage or ETA.

## Current evidence classification

The registry currently classifies 14 design decisions:

- 3 supported within a bounded scope;
- 8 partially supported because the paper setting differs materially;
- 2 supported only by internal exploratory experiments;
- 1 explicit hypothesis: that the smallest effective guidance fragment is generally preferable.

The counterbalanced internal experiment is especially important. Across eight probes, only 6 Sol,
3 Terra, and 4 Luna fixed-order modes remained the same unique mode after complete rotations. This
does not isolate letter bias causally, but it proves that fixed-order stability is too weak to gate
prompt calibration.

When both opposing user profiles were counterbalanced, they still produced different unique modes
in 7/8 Sol, 6/8 Terra, and 5/8 Luna families. This supports user conditioning within those measured
families while correcting the stronger fixed-order 8/8, 8/8, 5/8 result. Several condition-level
series were not exactly stable, so a unique mode is not automatically a production-ready policy.

Two later family-novel checkpoints added four preference families with complete four-position
rotations. The first replicated profile-sensitive decision ownership across all three models but
found less uniform review-report adaptation in Luna. In the second, all three models were exactly
stable under both profiles on requirements-artifact formality and context compactness: the
speed/autonomy profile selected direct compact action, while quality/control exposed trade-offs and
user choice. This extends the internal evidence beyond the original eight families, but it still
measures draft forced choices rather than downstream development quality and does not validate a
universal two-profile representation.

A third checkpoint first rejected two draft families without provider calls because an unsupported
probability claim and a frozen-plan action confounded preference with normative quality. After a
hash-bound revision, all three models chose higher upside with rollback under speed/autonomy.
Quality/control moved toward explicit user risk ownership and deeper advance planning, although
Sol's risk choices remained dispersed and Luna's fast-planning choices tied. This adds two families
while reinforcing that profile responsiveness and exact stability are separate requirements.

## Rules derived from the evidence

1. Preserve raw events and concrete actions; use aggregates for navigation.
2. Counterbalance MCQ actions across labels and map displayed answers back to canonical actions.
3. Repeat conditions and preserve ties; never infer a trait from one response.
4. Keep choice-only and self-report conditions separate.
5. Condition preference-sensitive behavior on the active user's priorities.
6. Search prompts automatically only against an immutable baseline and declared objective.
7. Require held-out family-atomic validation, normative non-regression, and latency limits before
   activating guidance.
8. Label unreviewed probe claims and product design preferences as hypotheses rather than borrowing
   certainty from adjacent papers.
9. Report actual model/tool lifecycle phases during waits, but never manufacture percent-complete
   or ETA values for non-linear agent work.
10. Treat model feedback as a source of artifact-grounded hypotheses and paired experiments, never
    as an automatic instruction to rewrite the harness.

The 4 KiB per-role runtime ceiling is a conservative product safety bound against accidental prompt
inflation, not evidence that 4 KiB is optimal. Candidate fragments must still demonstrate held-out
gain over the unchanged prompt and report measured input-token and latency cost; the ceiling cannot
substitute for that comparison.

This evidence map is intentionally conservative. A citation justifies only the scope stated in the
registry; it is not permission to claim that a technique is universally superior.
