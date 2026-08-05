# Exploratory behavior study: GPT-5.6 Sol, Terra, and Luna

## Evidence boundary

This is an exploratory report over draft probes and one catalog revision. Most conditions have one
repetition; the no-profile choice-only baseline has three. It describes observable choices and does
not claim access to private
chain-of-thought and does not establish stable model traits. Numeric summaries locate patterns,
but the selected actions in the linked reports are the primary evidence.

All 678 historical responses in this report used fixed action-to-letter order. They therefore cannot
separate action preference from option-ID or position bias. Subsequent runs use deterministic
balanced rotations and preserve both provider letter and canonical action; historical conclusions
must be rechecked under that protocol before prompt calibration.

The study produced 1,182 valid provider responses and one preserved transport-diagnostic failure:

- 3 smoke responses: one identical isolated normative question per model.
- 90 normative responses: 30 identical questions per model.
- 90 preference responses: 10 identical questions per model under three conditions: no user
  profile, fast/autonomous user, and quality/control-oriented user.
- 30 isolated self-report responses: the same 10 no-profile preference questions per model,
  requesting a brief criterion, missing context, and verbal confidence.
- 60 additional choice-only responses: repetitions 1 and 2 of the 10 no-profile preference
  questions for each model.
- 87 category-stratified raw preference responses: 29 new questions (one per category) for each
  model, selected by a dataset-bound deterministic manifest.
- 174 additional category-stratified responses: repetitions 1 and 2 of those 29 questions for each
  model.
- 144 targeted profile responses: eight predeclared raw-prior families, three repetitions per
  model under speed/autonomy and quality/control user profiles.
- 96 counterbalanced raw-prior responses: the same eight targeted families with every canonical
  action rotated once through A, B, C, and D for each model.
- 192 counterbalanced profile responses: four complete rotations under speed/autonomy and
  quality/control for the same eight families and three models.
- 24 category-stratified v2 checkpoint responses: two previously unseen families with all four
  action positions for each model.
- 48 opposing-profile checkpoint responses: both v2 families under speed/autonomy and
  quality/control, with four rotations per model and condition.
- 24 category-stratified v2 checkpoint 2 responses: two further unseen families with all four
  action positions for each model.
- 48 opposing-profile checkpoint 2 responses: both new families under speed/autonomy and
  quality/control, with four rotations per model and condition.
- 24 category-stratified v3 checkpoint 3 responses: two further family-novel probes with all four
  action positions for each model, after a no-call human preflight revision.
- 48 opposing-profile checkpoint 3 responses: both new families under speed/autonomy and
  quality/control, with four rotations per model and condition.
- 0 response errors, 0 rate limits, and 0 automatic retries after the transport repair.
- 76 unique probes exercised out of the current 684 (approximately 11.1% unique-probe coverage).
  Repeated conditions are deliberately counted as responses, not as new questions.

Every request used a fresh conversation, no system message, no prior-question history, one active
request at a time, and a minimum two-second start interval. The only protocol instruction asked
for one JSON choice. Preference conditions added the user's natural-language priorities to the
same user message. All three models used reasoning effort `medium` and catalog revision
`e811fab79c2af26e14e89f2ded6190ed`.

## What the normative sample shows

Sol, Terra, and Luna selected the draft normative key on all 30 questions and made identical
choices on all 30. This supports a narrow conclusion: all three recognized the basic safe or
evidence-preserving action in this sample.

It does not show that the models behave identically. The distractors in this sample were too easy,
creating a ceiling effect. These probes are useful as minimum-capability regression gates, but not
as the main calibration instrument. The next normative revision should use plausible competing
actions, incomplete evidence, and trade-offs where the best action turns on one decisive detail.

Primary evidence: [sample30 comparative report](sample30.comparative-report.md).

## Raw preference priors without a user profile

The first repetition showed agreement on 6 of 10 preference questions and divergence on 4. After
three isolated repetitions, Sol was exactly stable on 8/10 questions, Terra on 7/10, and Luna on
8/10. Modal choices agreed across models on 7/10 questions. This replication materially changes
how the original divergences should be read:

- Vague visual cleanup is the strongest observed model difference: Sol and Terra consistently
  applied the nearest established convention and showed the diff, while Luna consistently chose
  the smallest convention-aligned cleanup and invited expansion.
- Ambiguous local naming remained different in modal behavior: Sol consistently implemented and
  disclosed the nearest analogous convention; Terra consistently produced a small locally dominant
  draft and invited correction. Luna's mode matched Sol, but Luna changed once, so its prior is less
  certain.
- The initial long-context difference did not replicate as a modal difference. All three modes chose
  a compact summary with on-demand expandable chronology; Sol and Terra each selected another
  artifact once.
- The initial claim that Luna preferred sequential ambiguity questions was unstable. Luna selected
  three different actions across three repetitions, while Sol and Terra consistently proposed
  defaults and requested one batched confirmation.

Across the other questions modal choices converged on concise implementation/reporting, reversible
progress under uncertainty, and user-selected review cadence. A three-run mode is still preliminary,
but it is a safer calibration input than any single response.

Primary evidence: [repeated-choice stability report](raw-prior.stability-report.md) and
[repetition-zero cross-model report](raw-prior.comparative-report.md).

## Category-stratified raw preferences

A second raw-prior sample added one previously unseen preference probe from every one of the 29
categories, then repeated each three times. Exact within-model stability was 17/29 for Sol, 19/29
for Terra, and 21/29 for Luna; cross-model modal choices agreed on 13/29. Only one divergent family
was exactly stable in all three models: Sol and Terra defined contracts across all requirement
slices and implemented incrementally, while Luna fully specified one slice and used feedback before
elaborating the next.

The concrete actions also show why first responses cannot be treated as traits:

- Sol often made bounded progress without another user turn, including prototyping a vague workflow
  and switching directly to an equivalent fallback evidence channel.
- Terra often exposed a decision interface first, including measured implementation alternatives,
  fallback trade-offs, and quantified pilot risk.
- Luna sometimes used consequence-sensitive escalation, selectively re-reading or widening tests,
  while also preferring to finish a safe atomic step before honoring an interruption.
- Sol's initial testing choice changed to the same modal focused-plus-impacted stopping rule as
  Terra; Luna remained stable on provisional delivery followed by a non-blocking full suite.
- Sol's initial preference for silence during a long-running build changed twice to a heartbeat,
  matching Terra and Luna modally.

Three repetitions expose instability but still do not establish population-level traits. The
stable and partially stable differences are useful for choosing explicit-profile tests next, but
not yet for emitting model-specific production prompts.

Primary evidence: [category-stratified qualitative analysis](CATEGORY_STRATIFIED_ANALYSIS.md),
[all 29 concrete comparisons](category-stratified-v1.comparative-report.md), and the
[repetition-by-repetition stability report](category-stratified-v1.stability-report.md), plus the
[frozen manifest](category-stratified-preference-v1.manifest.json).

### Category-stratified v2 checkpoint

A family-aware campaign excluded all 68 previously observed decision families rather than merely
their 70 probe IDs. Twenty-six categories still had an unseen preference family; ambiguity and
clarification, long-context position, and vague-requirement analysis had none. The shortfalls are
recorded in the manifest instead of being hidden by sibling-variant selection.

The first checkpoint ran two unseen families with complete four-position rotations. Sol always
reported code-review findings as blocker-first prose; Terra did so three times and used a
severity/effort table once; Luna used the table three times and blocker-first prose once. For a
complex architecture choice whose remaining weights belong to the user, Terra always presented the
Pareto frontier and requested the decisive weight, while Luna always recommended from the stated
profile and invited correction. Sol split evenly between those two actions.

These are raw priors, not rankings. The decision-ownership divergence is a strong candidate for an
opposing-profile treatment: some users want the final weight returned to them, while others prefer
the agent to turn an existing profile into a recommendation. The code-review difference similarly
motivated a concise-reading versus audit/control treatment.

That predeclared treatment added 48 counterbalanced responses. For decision ownership, all three
models moved toward a reversible default with later measured review under speed/autonomy and toward
showing the Pareto frontier plus asking the user for the decisive weight under quality/control.
Sol and Terra also moved from blocker-first concise review reporting under speed/autonomy to an
interactive category walkthrough under quality/control. Luna shared the quality/control pattern but
did not establish a stable distinct fast-profile mode.

The raw model differences are therefore better explained as priors that explicit user context can
override than as permanent model traits. This supports compact runtime preference context and
argues against adding broad model-specific behavior instructions to the system prompt. These two
families are still draft, so this evidence can guide further validation but cannot deploy a prompt
profile by itself.

Primary evidence: [checkpoint qualitative analysis](CATEGORY_STRATIFIED_V2_CHECKPOINT1_ANALYSIS.md),
[every action and presentation mapping](category-stratified-v2.checkpoint1.stability.md), and the
[profile-treatment action report](category-stratified-v2.checkpoint1.profile-adaptation.md), plus the
[family-aware campaign manifest](category-stratified-preference-v2.manifest.json).

The second checkpoint tested requirements-artifact formality and context compactness. Raw priors
differed: Sol always used a compact requirements checklist while Terra and Luna always exposed
artifact alternatives; context delivery was unstable in every model. Explicit priorities produced
the same exactly stable separation in all three models. Speed/autonomy always selected the compact
checklist and supplied only the nine decisive anchored excerpts. Quality/control always exposed
artifact maintenance costs or context token/omission trade-offs and let the user choose.

This is the cleanest checkpoint-level evidence so far that a small user-objective treatment can be
more useful than model-specific fixed instructions. It supports adaptive context and interaction
policies, but only for the tested draft families; it does not establish a universal profile axis.

Primary evidence: [checkpoint 2 qualitative analysis](CATEGORY_STRATIFIED_V2_CHECKPOINT2_ANALYSIS.md),
[raw actions and mappings](category-stratified-v2.checkpoint2.stability.md), and the
[profile-treatment action report](category-stratified-v2.checkpoint2.profile-adaptation.md).

The third checkpoint added reversible risk posture and planning depth after rejecting its original
v2 shard without making provider calls. Preflight had found unsupported probability language and a
frozen-plan action that would ignore invalidating evidence; v3 repaired both family variants before
freezing a new campaign.

Under speed/autonomy, every model chose higher upside with a rollback trigger for risk. Sol and
Terra always chose a compact dependency-ordered plan; Luna split between that and a first verified
slice. Under quality/control, Luna always returned risk appetite to the user while Terra did so
three times and Sol twice. Deeper advance planning was the quality/control mode for every model,
exactly stable in Luna and three-to-one in Sol and Terra.

The checkpoint again supports runtime objectives over fixed model stereotypes, while showing why
ties and residual instability must remain visible. It also establishes a practical campaign gate:
draft preference options require human normative preflight before subscription calls.

Primary evidence: [checkpoint 3 qualitative analysis](CATEGORY_STRATIFIED_V3_CHECKPOINT3_ANALYSIS.md),
[raw actions and mappings](category-stratified-v3.checkpoint3.stability.md), and the
[profile-treatment action report](category-stratified-v3.checkpoint3.profile-adaptation.md).

## Elicitation changes the measured decision

Requesting a brief externally reportable criterion changed 5 of 10 Sol choices, 3 of 10 Terra
choices, and 3 of 10 Luna choices relative to isolated choice-only calls. This is not harmless
metadata collection: the measurement protocol alters the behavior being measured.

Two induced changes were shared by all three models:

- For four unrelated ambiguities, every model moved to one structured message containing four
  concise questions and recommended defaults.
- For the long-context artifact, every model moved to showing token-cost and omission/audit
  trade-offs and asking the user to choose.

The expressed criteria repeatedly cited the absence of an active user preference. Because the
self-report schema explicitly asks for missing context, it appears to prime models toward surfacing
that absence and selecting more deliberative or user-controlled actions. Sol also added explanation
or teaching in two interaction questions and moved a stakeholder decision toward a provisional
reversible choice with a feedback window.

Median verbal confidence remained high (Sol 0.91, Terra 0.84, Luna 0.90) despite these protocol
changes. Therefore verbal confidence is not evidence that a choice is protocol-stable, and it must
not be used as a substitute for repeated behavioral observations.

Practical consequence: use choice-only calls as the primary decision baseline. Keep self-reported
criteria as a separate diagnostic channel for prompt authors; never merge the two conditions or
interpret the explanation as faithful private reasoning.

Primary evidence: [choice-only versus self-report report](elicitation.comparative-report.md).

## Adaptation to different users

Changing only the user's priorities changed 6 of 10 choices in Sol, 9 of 10 in Terra, and 6 of 10
in Luna. In this sample Terra was the most profile-responsive. That is a behavioral observation,
not a quality ranking: high responsiveness can be useful personalization or excessive sensitivity,
depending on whether the resulting action remains safe and effective.

The clearest shared adaptations were:

- Naming ambiguity: the fast profile caused immediate bounded progress or a draft; the
  quality/control profile caused comparison or a focused question before editing.
- Vague visual cleanup: the fast profile caused convention-aligned implementation; all three used
  two small mockups and user selection for the quality/control profile.
- Ambiguity batching: all three proposed defaults plus one batched confirmation for the fast
  profile. Under quality/control, Sol and Terra asked sequentially; Luna used a compact comparison
  table and allowed partial answers.
- Long context: fast-profile choices favored anchored or expandable summaries. Under
  quality/control all three exposed token, omission, and audit trade-offs and let the user choose.
- Explanation depth: Terra and Luna moved from concise outcome/evidence to walkthroughs under the
  quality/control profile. Sol also became more control-oriented elsewhere, but did not change this
  sampled explanation-depth choice.

Model-specific action evidence:

- [Sol across all three conditions](sol.three-profile-report.md)
- [Terra across all three conditions](terra.three-profile-report.md)
- [Luna across all three conditions](luna.three-profile-report.md)
- [Fast/autonomy cross-model comparison](fast-autonomy.comparative-report.md)
- [Quality/control cross-model comparison](quality-control.comparative-report.md)

### Replicated targeted profile experiment

A later experiment predeclared eight raw-prior families and repeated each three times under the two
profiles. Sol and Terra had different unique modes between speed/autonomy and quality/control on all
eight questions. Luna separated on five; two quality/control families had no unique mode and its
verification mode remained unchanged.

The most important finding is convergence under an explicit objective: all three models chose the
same profile-dependent action for implementation boundaries, tool-error fallback, and vague-workflow
prototyping versus clarification. All three also selected full-suite verification under
quality/control. This weakens any attempt to encode broad permanent model stereotypes in system
prompts; runtime user priorities often explain more of the desired behavior.

Residual differences remained in multi-slice requirement iteration and independent-verification
depth. Those should be calibrated as narrow decision families, not as a global model score.

Primary evidence: [targeted qualitative analysis](PROFILE_TARGET_ANALYSIS.md),
[all repetition-level profile actions](profile-target-v1.adaptation-report.md), and the
[predeclared manifest](profile-target-v1.manifest.json).

### Option-order correction

The eight targeted raw-prior families were rerun with complete four-way option rotations. Only 6/8
Sol, 3/8 Terra, and 4/8 Luna fixed-order modes survived as the same unique counterbalanced mode.
Other families changed or became tied. Counterbalanced exact stability was 4/8 for Sol, 2/8 for
Terra, and 3/8 for Luna.

This materially downgrades every fixed-order stability and profile-adaptation conclusion. It does
not prove that displayed letter alone caused each change—the sample count and run time also differ—
but fixed-order results can no longer gate prompt calibration. New runs use deterministic balanced
rotations and store provider letter, presentation mapping, and canonical action separately.

Primary evidence: [option-order qualitative analysis](OPTION_ORDER_ANALYSIS.md),
[every fixed/balanced action and mapping](profile-target-v1.option-order-report.md), and
[counterbalanced stability](profile-target-v1.counterbalanced-stability.md).

### Counterbalanced profile correction

After rotating both user profiles, fast/autonomy and quality/control retained different unique
canonical modes in 7/8 Sol, 6/8 Terra, and 5/8 Luna families. The fixed-order experiment had reported
8/8, 8/8, and 5/8. Explicit user priorities therefore remain a strong treatment, but fixed ordering
overstated its breadth for Sol and Terra.

Implementation boundary and vague-workflow behavior were exactly stable and profile-sensitive for
all three models. Full-suite testing was also the exactly stable quality/control action for all
three. Other families had unique modes without exact stability or unresolved ties, so they require
more evidence before prompt calibration.

Primary evidence: [counterbalanced profile analysis](COUNTERBALANCED_PROFILE_ANALYSIS.md),
[all canonical actions and repetitions](profile-target-v1.counterbalanced-adaptation.md), and the
fixed-versus-balanced [fast](profile-target-v1.fast-autonomy.option-order.md) and
[quality](profile-target-v1.quality-control.option-order.md) reports.

## Model-assisted question triage and revision

Question quality was evaluated in a separate channel from behavioral-choice observations. Each
call received one complete two-variant family in a fresh conversation, with author answers,
rationales, and generator identity hidden. These diagnostics do not count among the 966 behavioral
responses and have no authority to approve or mutate a probe.

The first 24 valid triage responses exposed a flaw in the audit protocol itself. Reviewers were not
told that active user preferences arrive only in profiled conditions, or that paired questions are
semantic-equivalence robustness replicates rather than single-factor causal experiments. This
caused systematic false warnings about missing preferences and deliberate paraphrase/reordering.
Those results remain preserved as methodological evidence and are explicitly barred from driving
question edits. Eight earlier DNS failures are preserved separately and contain no model response.

Protocol v2 stated the instrument contract and effect-axis meanings without revealing author
labels. Across 24 corrected responses, all three models passed implementation reuse, recovery
fallback, requirements iteration, and verification independence. Concrete multi-model findings
remained in three families:

- interruption variants said the entire operation took eight seconds while an action assumed eight
  seconds remained (Sol and Luna);
- test-scope variants changed impacted coverage, a quantified 2% flake rate, and release-gate facts
  rather than merely paraphrasing them (Sol and Terra);
- a static uncertainty table claimed interaction and user control despite introducing no exchange
  or checkpoint (Sol and Terra).

Those findings produced a hash-bound v2 draft with exactly six changed probes. IDs, families,
evaluation modes, review status, answers, and reviewer fields are unchanged. The original dataset
hash remains `c60609286bb191ee3266d314f10d334c576f63cee1f02e4cbc58075e3769f514`; the revised
draft hash is `ba8aa4640012638590ea190325dfc7f07d7a20a5cc0e7abe0c3fd0f8dfb6eea0`.

A predeclared nine-response post-edit check then sent the three revised families once to each model.
Terra and Luna passed all three. Sol confirmed the original defects were gone but proposed three
different effect/dominance revisions; because no second reviewer shared any of those signals, they
remain review candidates rather than automatic edits. The v2 dataset is still entirely draft and
requires independent human, family-atomic review.

Primary evidence: the corrected [24-response qualitative report](profile-target-v1.triage-v2-report.md),
the [post-edit verification report](profile-target-v1.triage-v2-revisions-report.md), and the
[v1 methodological diagnostic](profile-target-v1.triage-v1-methodology.md). Dataset changes are
reproducible from `../../model_behavior_probes.v2.revision.json` and traced in
`../../model_behavior_probes.v2.lineage.json`.

## Implications for Infinidev prompt calibration

The evidence favors a layered policy, not one magic system prompt:

1. Keep invariant safety and evidence-handling guidance small and shared. The normative sample did
   not expose a model-specific deficiency that justifies extra prompt text yet.
2. Represent user priorities explicitly at runtime. The same model often chose materially different
   actions when speed/autonomy versus quality/control was stated.
3. Calibrate by concrete decision family. Interaction cadence, ambiguity timing, explanation depth,
   and long-context artifacts shifted differently; one global "more autonomous" scalar would erase
   useful distinctions.
4. Preserve model priors when they already match the user. Add guidance only where repeated held-out
   evidence shows a mismatch, then verify that the change does not damage normative behavior.
5. Keep raw selected actions and replies as the source of truth. Axis scores can rank candidate
   policies, but prompt authors should read the before/after actions that produced those scores.

## What remains unproven

- The no-profile baseline has only three repetitions, while normative and explicit-profile
  conditions still have one; none supports a population-level stability estimate.
- All probes remain draft. Model-assisted triage improved six questions but is not independent
  approval; normative keys and preference effects still need blind, family-atomic human review.
- Self-report criteria now cover only 10 preference probes and showed substantial protocol effects;
  their stability and usefulness on other categories remain unproven.
- Only 70 of the current 684 unique probes have live coverage. Every category now has one raw
  preference probe with three repetitions per model, but this is still only one family per category.
- The model catalog can change; these results apply only to the recorded identities and revision.

No production system prompt should be auto-generated from this sample alone. The next evidence gate
is replicated, category-stratified coverage plus isolated choice-only/self-report comparison on a
held-out subset.
