# Category-stratified raw preference analysis

## Evidence boundary

This report analyzes 261 fresh provider responses: the same 29 preference probes, one from every
dataset category, run three times against GPT-5.6 Sol, Terra, and Luna. Every call was an isolated fresh
conversation with no system message or prior-question history. Calls were single-flight, used a
minimum two-second start interval, had no automatic retries, and stopped on rate limiting. The run
had 261 valid responses, zero errors, and zero rate limits.

These are unprofiled preference probes, so there is deliberately no universal correct answer.
Three repetitions can expose obvious instability but still cannot estimate population-level model
traits. The primary evidence is the concrete selected action in the
[full comparative report](category-stratified-v1.comparative-report.md), not the aggregate counts
below.

The frozen [selection manifest](category-stratified-preference-v1.manifest.json) binds all 29 probe
IDs and their category metadata to dataset SHA-256
`c60609286bb191ee3266d314f10d334c576f63cee1f02e4cbc58075e3769f514`.

## Replication result

Exact within-model stability was 17/29 for Sol, 19/29 for Terra, and 21/29 for Luna. Cross-model
modal choices agreed on 13/29 questions. This means 30 of the 87 model-by-question series changed at
least once; a single response would have been an unsafe prompt-calibration input in more than one
third of the measured series.

Only one cross-model difference was exactly stable in all three models: for a five-slice complex
requirement, Sol and Terra defined hard contracts for every slice, left reversible presentation
choices open, and implemented incrementally; Luna fully specified and implemented slice one, then
used feedback before elaborating later slices.

Several repetition-zero impressions did not survive unchanged:

- Sol's initial silence during a long build was not stable. Its modal choice, like stable Terra and
  Luna, was to send a concise progress heartbeat.
- Sol's initial dependency-selected test subset changed twice to accepting focused plus impacted
  tests, matching stable Terra modally. Luna remained stable on provisional delivery followed by a
  non-blocking full suite.
- Terra's initial draft-and-invite naming choice changed twice to selecting the nearest analogous
  convention and disclosing it, producing modal agreement across all three models.
- The initial three-way agreement on fixed-budget code review did not hold: Luna changed twice to a
  risk-ranked review with broader treatment only for high-risk surfaces.
- The initial three-way uncertainty choice did not hold: Sol changed twice to a recommendation with
  calibrated confidence and explicit evidence that would change it; Terra and Luna stayed with a
  concise confidence qualifier plus tested fallback.

The complete repetition-by-repetition evidence is in the
[stability report](category-stratified-v1.stability-report.md). The sections below retain the first
pass as hypothesis-generation context; this replication result supersedes any conflicting claim.

## Repetition-zero shared raw defaults

On repetition zero, the three models selected the same action on 14 of 29 questions. The common
choices are more informative as behavioral defaults than as scores:

- They exposed trade-offs and returned control to the user for context refresh policy, fixed-budget
  review depth, evidence-channel choice, destructive cleanup redundancy, and whether to include a
  nearby cleanup with a precise edit.
- They preferred reversible progress when a newer instruction could supersede older wording, and
  they chose a bounded nearby typo/comment cleanup when authorization was already clear.
- They favored semantic or wiring-aware codebase investigation first, with text search retained for
  fresh files that may not be indexed.
- They reported workspace preservation explicitly: task files, tests, no-commit status, and
  unrelated files left untouched.
- They used concise uncertainty qualifiers, cited bounded web evidence, and disclosed ignored prompt
  injection without letting the untrusted text redirect the task.
- Their interaction defaults were not uniformly autonomy-maximizing: all chose preview-then-correct
  for one reversible decision and checkpointed after each two-step verified planning milestone.

This shared behavior supports a compact common harness layer around evidence boundaries,
reversibility, workspace preservation, and explicit trade-offs. It does not support adding detailed
model-specific instructions in these 14 families.

## Repetition-zero candidate differences

Fifteen questions diverged on the first pass. They motivated replication, but must now be read
through the corrections above:

- **Sol often made bounded progress without another user turn.** It prototyped a vague workflow,
  selected the nearest naming convention and disclosed it, switched directly to a fallback search
  channel, recommended from three strong sources, and included an optional risk/experiment section
  in the completion handoff. Terra and Luna more often asked first or offered the extra material only
  on request.
- **Terra more often surfaced a decision interface before committing.** It asked the user to choose
  a fallback after seeing latency/equivalence trade-offs, prototyped competing implementation
  boundaries before selection, quantified pilot cost/upside/rollback before asking for risk posture,
  and used expandable evidence presentation. This may represent user-control sensitivity, but it
  may also create unnecessary interaction for autonomy-oriented users.
- **Luna sometimes used consequence-sensitive escalation.** It re-read source material only for the
  two highest-consequence long-context actions and expanded self-correction checks only when traced
  checks exposed coupling. On an interrupted atomic operation, however, Luna finished the safe
  atomic step before switching while Sol and Terra paused at the safe temporary-file boundary.
- **Testing behavior differed sharply.** Sol chose a dependency/co-change-selected cross-subsystem
  subset; Terra accepted focused plus impacted tests; Luna delivered provisionally and ran the full
  suite as a non-blocking follow-up. This is the clearest sampled family for calibrating speed,
  confidence, and completion semantics separately.
- **Progress communication differed.** Terra and Luna sent a concise heartbeat during a long build;
  Sol remained silent until completion. This maps directly to users who prefer visibility versus
  low-interruption execution.
- **Evidence independence differed.** Terra and Luna treated a hash plus deterministic regeneration
  as complete proof; Sol treated an independent check as optional. This should be tested under
  explicit auditability profiles before turning it into prompt guidance.

## What this means for prompt calibration

The raw choices reinforce that calibration should operate on decision families, not on a single
model score or a single “autonomy” slider. A useful prompt-authoring report should preserve, for each
family:

1. the scenario and all offered actions;
2. the model's actual selected action across repetitions;
3. the user's expressed priorities for that run;
4. whether a prompt intervention changes that concrete action;
5. any normative or safety regression caused by the intervention.

For example, a speed-oriented user may prefer Terra's focused-test stopping rule or Sol's bounded
fallback, while a user who values exhaustive final evidence may prefer Luna's provisional/full-suite
split. None is universally optimal. The harness should add the smallest relevant guidance only when
the replicated model prior conflicts with the active user's desired outcome.

## Next evidence gate

The 29 questions now have three isolated repetitions. The next evidence gate is to rerun the one
fully stable divergence and the strongest partially stable differences under explicit
speed/autonomy and quality/control profiles. Unstable families need more raw repetitions or revised
options before calibration. Draft families also still require independent, family-atomic review;
this run does not approve their wording or option design.
