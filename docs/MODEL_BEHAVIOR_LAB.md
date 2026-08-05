# Model Behavior Lab

Infinidev should adopt harness techniques only after a reproducible comparison, not because a
prompt sounds persuasive or one demo succeeds. This lab measures observable behavior; it does not
claim that a generated explanation reveals a model's private computation.

Methodological claims and their limitations are tracked in
[`MODEL_BEHAVIOR_EVIDENCE.md`](MODEL_BEHAVIOR_EVIDENCE.md) and the machine-audited
`bench/model_behavior_evidence.json`. Papers are evidence only for their measured scope; internal
experiments and product hypotheses are labeled separately.

## Experimental contract

Research starts from a concrete Infinidev–model problem, not from a desire to catalogue model
traits. For the problem-to-information-to-instrument chain and the current evidence-backed backlog,
see [`INFINIDEV_LLM_PROBLEM_MAP.md`](INFINIDEV_LLM_PROBLEM_MAP.md). Linguistic variables such as
register, lists, examples, `NEVER`, or `SOMETIMES` enter only when they can discriminate hypotheses
about a product problem.

There is no universal best prompt. The same safe, competent agent behavior can legitimately favor
more user control or more autonomy, more interaction or fewer interruptions, and speed or extra
quality assurance. The lab therefore preserves raw choices and separates two evaluation modes:

- `normative`: competence, safety, factual fidelity, explicit requirements, and permission
  boundaries have a defensible `answer` and `gold_rationale`.
- `preference`: several actions are acceptable, there is no universal `answer`, and every action
  has signed `choice_effects` on declared utility axes. A user-specific `utility_profile` determines
  which trade-off is preferable for that run.

Decision probes use `id`, `category`, `scenario`, `user_request`, `choices`, and
`evaluation_mode`, plus an `answer` for normative probes or `choice_effects` for preference probes.
Optional `group` and `tags` describe controlled families. A plain `prompt` remains supported for
established external datasets. The alternatives are concrete next actions, not personality labels.
A group contains meaning-preserving perturbations such as paraphrases, option reordering,
irrelevant context, or moving decisive context from the beginning to the middle. Normative variants
keep the correct decision fixed. Preference variants preserve the same underlying trade-off while
varying one presentation factor.

The initial coverage contract is 29 categories with at least 20 items each (580 items). Each
scenario family should then have two or three controlled variants, producing roughly 1,160–1,740
questions. `bench/model_behavior_taxonomy.json` is the machine-readable target. Audit a dataset:

```bash
uv run python -m bench.probe_dataset probes.jsonl bench/model_behavior_taxonomy.json
```

Every probe is assigned to `calibration` or `validation`. All variants in one `group` stay in the
same split; otherwise a paraphrase of a calibration example would leak into validation. Category
counts are necessary but not sufficient: review each category for easy/hard cases, consequential
and reversible actions, ambiguity, distractor quality, and coverage across supported model families.
Generated items begin with `review_status: draft` and do not count toward category targets.
Approval requires a named reviewer and a concise `gold_rationale`; rejected items remain in the
authoring artifact for audit but do not enter evaluation. The auditor also rejects normalized
duplicate questions. At release scale it additionally requires calibration/validation coverage,
exactly four actions, no normative answer position above 40% within a complete category, and two or
three approved variants per family.

Draft families can be generated in bounded batches with an explicitly identified author model:

```bash
uv run python -m bench.generate_probe_drafts \
  bench/model_behavior_taxonomy.json probes.jsonl \
  --model provider/author-model --generator-identity provider/author-model@revision \
  --category test_strategy --families 5 --variants 3
```

Generate preference-sensitive families separately so their contract cannot be confused with gold
label generation:

```bash
uv run python -m bench.generate_probe_drafts \
  bench/model_behavior_taxonomy.json preference-probes.jsonl \
  --model provider/author-model --generator-identity provider/author-model@revision \
  --category user_interaction --families 5 --variants 2 \
  --evaluation-mode preference
```

The preference authoring instruction requires four safe and competent choices, forbids a universal
answer, assigns effects for every choice, and requires at least two utility axes in tension. The
dataset audit reports global and per-category preference-axis coverage. These are structural checks,
not semantic approval: an independent reviewer must still reject fake trade-offs, dominated options,
effects that do not match the action, or variants that silently change the trade-off.

The generator currently authors normative drafts. It assigns whole families deterministically to
calibration or validation and refuses duplicate IDs. Generated answers and rationales are proposals,
not labels: an independent reviewer must inspect ambiguity, distractors, gold rationale, and whether
variants preserve the intended decision before changing `review_status` to `approved`. If review
finds multiple legitimate policies, convert the whole family to preference mode, remove `answer`,
and document the rationale and effects for every choice instead of forcing an arbitrary gold label.

### Independent blind review

The author cannot approve their own labels. Export a hash-bound packet that omits normative answers,
gold rationales, author analysis, and generator identity:

```bash
uv run python -m bench.probe_review export \
  bench/model_behavior_probes.draft.jsonl review-packet.json
```

The reviewer independently selects the normative answer or confirms that a preference probe has no
universal answer and that every declared effect matches its action. Each JSONL review row records the
immutable reviewer identity, packet dataset hash, verdict, evaluation mode, answer when normative,
rationale, and `effects_valid` when preference. See
`bench/model_behavior_reviews.example.jsonl`; replace its placeholder hash with the packet hash.

Create an auditable report and, only after it passes, a separate approved dataset:

```bash
uv run python -m bench.probe_review report \
  bench/model_behavior_probes.draft.jsonl reviews.jsonl review-report.json \
  --min-reviews 1
uv run python -m bench.probe_review apply \
  bench/model_behavior_probes.draft.jsonl review-report.json \
  model_behavior_probes.approved.jsonl
```

Approval is family-atomic: every controlled variant must have enough distinct independent reviews,
all reviewers must accept the proposed evaluation mode, normative reviewers must independently match
the author label, and preference reviewers must accept the effects. Self-review, one disputed
variant, a stale dataset hash, or a requested revision blocks the whole family. Use two reviewers
plus adjudication for consequential release sets; `--min-reviews 1` is the minimum mechanism, not a
claim that one opinion establishes objective truth. Drafts remain immutable evidence of authorship;
review produces a new artifact rather than silently rewriting labels.

Model-assisted triage is a separate, weaker diagnostic pass. It sends one complete two-variant
family per isolated call, hides author answers/rationales/generator fields, and changes displayed
action order deterministically per reviewer identity. It may identify overlapping actions,
dominated choices, effect mismatches, or variant leakage, but it has no path into
`probe_review apply`, cannot change `review_status`, and never counts as independent approval.
The blinded packet still states the instrument contract: preference profiles arrive as
natural-language user context in profiled experiments but are deliberately absent from the raw
baseline; paired variants are semantic-equivalence robustness replicates, so paraphrase and order
changes are expected while decision-relevant facts must remain stable; and effect-axis names describe
agent autonomy separately from user control. This metadata prevents the reviewer from mistaking a
deliberate evaluation condition for a question defect without exposing author labels or rationales.
Run each model separately with a raw one-condition configuration:

```bash
uv run python -m bench.run_probe_triage run \
  bench/model_behavior_probes.draft.jsonl sample.manifest.json \
  gpt-5.6-sol.raw.json sol-triage.jsonl

uv run python -m bench.run_probe_triage report triage-report.json \
  sol-triage.jsonl terra-triage.jsonl luna-triage.jsonl \
  --markdown triage-report.md
```

The triage runner enforces no system message, no utility profile, exactly one condition, and at
least 2 seconds between request starts. It shares the host-wide single-flight lock, disables
provider retries/cache, and stops after the first provider, transport, parse, or rate-limit error.
The Markdown report preserves each model's concrete finding and suggested edit; counts only show
where reviewers converged and are never an automatic decision rule. A human reviewer must inspect
proposed changes and review the resulting new dataset version independently.

The taxonomy explicitly separates planning, implementation strategy, testing, verification, code
review, local codebase investigation, web research, user interaction, complex decision-making,
decision support, and vague/complex requirement analysis. Composite families should cross two or
more categories because production failures often arise at their boundaries. Keep one primary
category for coverage accounting and express secondary dimensions as tags.

`bench/model_behavior_probes.draft.jsonl` currently contains 684 manually authored draft questions:
536 normative and 148 preference probes, organized as 342 two-variant families. This consists of
the 580-question base (20 in each of 29 categories) plus 104 newly materialized cross-category
preference probes. Each question carries an individual
hypothesis, decisive information, variant axis,
failure signal, and calibration use. Normative questions add a gold rationale and an explanation for
every distractor. Preference questions add the intended trade-off and a rationale plus effects for
every choice. Drafts remain excluded from release counts until an independent reviewer approves
them.

The original draft is retained as the immutable evidence base for historical manifests and runs.
Evidence-backed edits are expressed in `bench/model_behavior_probes.v2.revision.json` and
materialized as `bench/model_behavior_probes.v2.draft.jsonl`; the corresponding lineage file binds
both dataset hashes and lists every changed field, rationale, and source report. Reproduce it with:

```bash
uv run python -m bench.probe_revision \
  bench/model_behavior_probes.draft.jsonl \
  bench/model_behavior_probes.v2.revision.json \
  bench/model_behavior_probes.v2.draft.jsonl \
  bench/model_behavior_probes.v2.lineage.json
```

The revision tool refuses stale base hashes, unknown probes, repeated edits, missing evidence, and
changes to IDs, families, evaluation modes, review status, answers, or reviewer identity. A revised
draft is still a draft; lineage is not approval.

Draft campaign shards also need human normative preflight before any provider call. Every offered
action in a preference probe must remain a defensible policy under the stated facts; speed, control,
or autonomy cannot excuse false certainty, ignored evidence, unsafe behavior, or unauthorized side
effects. If preflight fails, preserve the rejected manifest, revise both semantic-family variants
into a new hash-bound dataset, and freeze a new campaign. Never repair an executed dataset in place
or count a rejected no-call shard as observed coverage. Revision v3 records the first application of
this gate in `bench/model_behavior_probes.v3.revision.json` and its lineage file.

Utility axes are deliberately small and observable: `autonomy`, `interaction`, `user_control`,
`speed`, `quality`, `cost_efficiency`, and `caution`. Effects and profile weights are bounded to
`[-1, 1]`; positive profile weights mean “prefer more of this axis” and negative weights mean
“prefer less.” Linear utility is an explicit, auditable first model, not a claim that human
preferences are intrinsically linear. Keep the original selected action so future scoring models
can reanalyze the raw data.

Global axis counts are insufficient: they can be satisfied by concentrating every legitimate
trade-off in interaction questions. The taxonomy therefore also requires at least four approved
preference probes (two controlled families) in every category. The dataset auditor reports authored
and approved shortfalls separately. `bench/preference_family_blueprint.json` originally planned 52
missing families/104 probes and records, for each family, the legitimate trade-off, hard-boundary assumption,
behavioral information sought, utility axes, and perturbation axis. Validate that plan against the
current dataset before authoring:

```bash
uv run python -m bench.preference_blueprint \
  bench/preference_family_blueprint.json bench/model_behavior_taxonomy.json \
  bench/model_behavior_probes.draft.jsonl
```

The blueprint passing does not make its future questions valid. It proves only that every current
category shortfall has an explicit, non-single-axis authoring target. Generated variants still begin
as drafts and require the same family-atomic independent review. All 52 families/104 probes in the
preference blueprint have now been materialized, so every category has at least two preference
families/four probes. The present 684-question artifact completes authored quantity and
cross-category preference coverage, but not independent review or approval.

Each observation has `probe_id`, `condition`, selected `answer`, `elicitation_protocol`, and
optional verbal `confidence` in `[0,1]`, `latency_seconds`, `tool_calls`, and `error`. A condition is
a frozen harness
configuration: model identifier and revision, system prompt hash, tool set, context policy,
temperature, and token budget should be stored with the run artifact. Do not store hidden
chain-of-thought. A short outcome rationale or cited evidence may be retained separately when the
provider permits it, but must not be treated as ground truth about internal reasoning.

Run analysis with:

```bash
uv run python -m bench.model_behavior \
  bench/model_behavior_probes.example.jsonl observations.jsonl \
  --baseline current \
  --utility-profile bench/model_behavior_utility_profile.example.json \
  --output validation-report.json
```

The report keeps normative and preference evidence separate. Normative rows produce accuracy,
Brier score, expected calibration error, perturbation success, and exact paired McNemar results.
Preference rows produce profile-conditioned mean utility, regret, paired utility deltas, and an
exact sign test. Both retain latency, tool calls, sample size, and errors; timeouts are never silently
discarded. One report belongs to exactly one utility-profile hash. Re-score the raw observations in
a separate report for another profile rather than averaging incompatible user objectives.

Numbers are comparison instruments, not the behavioral diagnosis used to write prompts. Build the
category-oriented qualitative dossier from the same immutable observations:

```bash
uv run python -m bench.behavior_dossier probes.jsonl observations.jsonl \
  --format markdown --output behavior-dossier.md
```

For every condition and category the dossier retains the selected action text, normative expected
action when applicable, expressed decision criterion, stated missing context, response excerpt,
confidence, errors, and the exact behavior of each perturbation-family variant. Normative failures
are linked back to their probe hypothesis, expected failure signal, and candidate calibration use.
Those links are prompt-authoring hypotheses, not automatically accepted prompt text: repeated
evidence suggests what guidance to try, and held-out evaluation decides whether it helps. Preference
decisions are reported as the actual policy selected, not collapsed into a utility number.

The raw observation JSONL remains authoritative and may retain the complete model response. Dossier
excerpts are bounded only to keep a report usable; every example carries a probe ID so an author can
return to the full response. Model-provided criteria and missing-context fields describe expressed
reasoning and feedback. They are valuable for understanding behavior, but do not prove that a
generated explanation faithfully exposes private internal computation.

Candidate authors should consume a JSON dossier through an evidence-first brief rather than being
shown only the leaderboard:

```bash
uv run python -m bench.behavior_dossier probes.jsonl observations.jsonl \
  --format json --output behavior-dossier.json
uv run python -m bench.prompt_authoring_brief behavior-dossier.json \
  --condition current --output prompt-authoring-brief.md
```

The brief separates normative failures to address, successful behavior to preserve, preference
behavior that must remain profile-conditioned, unstable perturbation families, the model's expressed
criteria, and stated missing context. A prompt author must cite those records when proposing a small
candidate fragment. A single surprising response is retained but marked as a hypothesis; repeated
patterns carry stronger authoring weight. Candidate generation never edits the production prompt
directly—the unchanged baseline and every proposal still go through calibration and held-out
validation.

## Running probes

Define the immutable model identity and candidate prompts in a run configuration (see
`bench/model_behavior_run.example.json`). Credentials stay in the provider's environment; they are
never written to observations. Preference probes require a named `utility_profile` in that
configuration; its identity and hash are written to every observation. Run calibration and
validation independently:

```bash
uv run python -m bench.run_model_behavior probes.jsonl run.json calibration.jsonl \
  --split calibration --repetitions 3
uv run python -m bench.run_model_behavior probes.jsonl run.json validation.jsonl \
  --split validation --repetitions 3
```

For a comparable category-stratified study, freeze the sample before contacting any model:

```bash
uv run python -m bench.probe_manifest probes.jsonl sample.manifest.json \
  --evaluation-mode preference --per-category 1 --seed 20260803 \
  --exclude-observations prior-observations.jsonl
uv run python -m bench.run_model_behavior probes.jsonl run.json observations.jsonl \
  --manifest sample.manifest.json --include-drafts --allow-unprofiled-preferences
```

The manifest records the exact dataset SHA-256, selected IDs, category/family metadata, seed, and
prior-coverage exclusion count. The runner rejects a changed dataset, duplicate or missing IDs, and
conflicting selection filters. This prevents silent question drift between separate model runs.
Every new observation also records both `dataset_sha256` and `manifest_sha256`; these fields are
part of the resume identity, so retaining a probe ID across a question revision cannot silently
reuse the older answer. Pre-field artifacts can be bound only through
`bench.observation_provenance`, which refuses conflicting hashes and emits source-to-bound artifact
lineage without changing raw responses.

For coverage campaigns, exclude observed families rather than only observed IDs; otherwise a
semantic sibling can inflate unique-question counts without adding a new decision family. If some
categories have no unseen family left, require an explicit partial-campaign declaration so the
shortfalls remain visible:

```bash
uv run python -m bench.probe_manifest probes.jsonl campaign.manifest.json \
  --evaluation-mode preference --per-category 1 --seed 20260804 \
  --exclude-observed-families --allow-category-shortfalls \
  --exclude-observations prior-*.observations.jsonl
```

Counterbalanced campaigns must checkpoint by whole probes, never by individual shuffled calls.
Freeze each checkpoint as a child manifest before contacting a provider. The child records its
parent manifest hash and a contiguous whole-probe range, so later checkpoints cannot silently drift
when an append-only output is resumed:

```bash
uv run python -m bench.probe_campaign campaign.manifest.json checkpoint-1.manifest.json \
  --start 0 --count 2
```

For a four-action catalog, this example runs the two frozen questions with all four rotations:

```bash
uv run python -m bench.run_model_behavior probes.jsonl run.json observations.jsonl \
  --manifest checkpoint-1.manifest.json --repetitions 4 \
  --include-drafts --allow-unprofiled-preferences
```

The runner rejects `--max-runs` under `balanced_rotation`, because even a numerically divisible
limit can cut incomplete cycles after work-item shuffling. `--max-probes` retains every configured
condition and repetition for each selected question and remains useful for exploratory runs, but a
material campaign should use explicit child manifests for durable checkpoint identity.

Use a condition such as `"raw": null` for the baseline behavioral study. This omits the system
message entirely; an empty or generic behavioral system prompt is not substituted. The only added
text is the response-format contract needed to parse and retain the answer. Calibrated prompt
conditions are separate experiments and must not be mixed into the raw baseline.

The raw baseline uses `"elicitation_protocol": "choice_only"`. It requests only
`{"answer":"A"}`: no confidence, explanation, decision criterion, or missing-context reflection is
requested in the same call. Those additions can induce a different mode of deliberation, so collect
them in a second run with `"elicitation_protocol": "self_report"`. That second run repeats each
question in a new conversation; it is not a follow-up and never sees the choice-only answer. Keep
the two protocols in separate observation/report artifacts. Agreement and disagreement between
them are behavioral evidence, while the self-report remains an expressed explanation rather than
hidden reasoning.

New MCQ runs should use `"option_order_protocol": "balanced_rotation"`. The runner applies a
deterministic cyclic permutation seeded by probe ID and repetition, maps the provider's displayed
letter back to the canonical action key, and stores both the displayed letter and mapping. A complete
cycle requires repetitions divisible by the number of choices (four repetitions for the current
four-action probes); the runner rejects partially balanced claims. Historical fixed-order artifacts
remain valid observations of the presented questions but cannot separate action preference from
option-ID or position bias.

This control follows the empirical warning in [Large Language Models Are Not Robust Multiple Choice
Selectors](https://arxiv.org/abs/2309.03882): option-position changes can materially alter MCQ
answers. Counterbalancing reduces that confound; it does not prove that every remaining choice is a
stable underlying preference.

Compare the two isolated artifacts without collapsing their raw responses:

```bash
uv run python -m bench.compare_elicitation \
  choice-only-observations.jsonl self-report-observations.jsonl \
  elicitation-comparison.json
```

The comparator requires identical model revision, condition hash, utility-profile hash, probe ID,
and repetition. It reports answer agreement, unpaired observations, every changed selection, both
complete responses, and the self-reported criterion/context. A disagreement is not automatically an
error; it demonstrates that the elicitation protocol changed observable behavior.

For repeated profile experiments, compare raw, speed/autonomy, and quality/control artifacts with
the concrete-action report:

```bash
uv run python -m bench.profile_adaptation_report probes.jsonl adaptation.md adaptation.json \
  --model Model=raw.jsonl,fast.jsonl,quality.jsonl
```

The reporter requires matching model identities, probe sets, repetition sets, and the expected
profile names and hashes. It retains every repetition-level action and raw response, marks exact
within-condition stability, and reports a profile change only when both conditions have unique
modes. Aggregate change counts are navigation aids, not utility or quality rankings.

Preference probes insert the profile's natural-language description into the user message as part
of the scenario. Internal utility-axis weights are retained for later analysis but are not shown to
the model. This preserves the user's actual requested trade-off as the decision context instead of
teaching the model to optimize the evaluator's numeric representation.

Runs are deliberately single-flight. Each provider call receives a newly constructed `messages`
list containing only the current probe (and its system message only for a non-raw condition); no
conversation or prior answer is carried forward, and provider caching is disabled. Calls execute
one at a time, including across any two local behavior-runner processes, even if they target
different providers or models. `min_request_interval_seconds` controls pacing
and defaults to 2 seconds between request starts. LiteLLM internal retries are disabled, and the run
stops immediately after recording a 429/rate-limit response so resuming cannot create a retry
storm. This host-local lock cannot coordinate runners on different machines, so do not run
experiments from multiple hosts concurrently.

### ChatGPT subscription routes

The runner accepts `"provider": "openai_subscription"` and reuses Infinidev's shared Codex OAuth
transport. It resolves or refreshes the credential for every request, normalizes the model to the
Responses route, targets the Codex subscription backend, supplies the account/client headers, and
sets server-side `store` to false. Credentials are never copied into the run configuration or
observation JSONL. See `bench/model_behavior_run.openai_subscription.example.json`; run Sol, Terra,
and Luna as three separate commands/configurations so the global single-flight lock and per-model
artifacts remain obvious.

The selectable slugs must come from the logged-in account's live Codex catalog, not from a guessed
family name. Bind `model_identity` to the exact slug plus the catalog ETag/fetch revision and record
`reasoning_effort`; changing either creates a different experiment. Never infer that one variant is
available merely because another is available.

This route has a support-boundary caveat. OpenAI publicly documents using Codex with eligible
ChatGPT plans and programmatic control through Codex, but the internal subscription backend and
the Sol/Terra/Luna slugs are not a documented general-purpose API contract. The live account catalog
is evidence of current technical availability, not a guarantee that bulk third-party evaluation is
supported or exempt from abuse controls. Therefore:

1. require explicit operator approval before the first real subscription call;
2. smoke-test one question per model, sequentially;
3. stop on the first authentication, entitlement, policy, or 429 response without retrying;
4. expand in small checkpoints only after inspecting usage/account health;
5. use the metered public API instead when contractual stability and reproducibility outweigh
   subscription cost.

Each call is appended immediately. Re-running the command skips the same
probe/condition-hash/repetition, while changing a prompt hash schedules it again. `--category`
supports staged runs and `--max-runs` supports smoke tests. Observations retain the original model
response, and—only under `self_report`—short decision criterion, self-reported missing context, and
verbal confidence, plus latency, token counts, model identity, prompt hash, and protocol. Under
`choice_only`, those self-report fields remain absent rather than being fabricated as zero. Provider
errors and malformed JSON remain explicit rows.
The short criterion and missing-context fields are model self-reports. Analyze their recurring
content, contradictions, and changes across controlled variants; do not merely count whether the
fields are present. They remain useful for hypothesis generation rather than evidence of private
reasoning.
Only approved probes execute by default. Exploratory runs over authored content must pass
`--include-drafts`, and their observations cannot be used for a deployment profile until the
underlying questions are independently approved.

### 2026-08-04 exhaustive raw and counterbalanced campaign

The first complete exploratory campaign ran all 684 authored drafts independently against the live
subscription catalog identities for GPT-5.6 Sol, Terra, and Luna: 2,052 fixed-order responses. The
models agreed on 606 probes and diverged on 78. Because a fixed presentation confounds action content
with position, those 78 divergences were frozen in a dataset-bound manifest and rerun with all four
cyclic rotations: 312 calls per model and 936 calls total. Every call used a fresh conversation, no
system message, choice-only elicitation, one active request, at least two seconds between starts, no
provider retry, and immediate durable output. All 936 completed without an error or missing answer.

The counterbalanced evidence materially changes how the original answers should be interpreted:

- Sol was exactly stable on 24/78 probes, Terra on 26/78, and Luna on 29/78.
- Only 23/78 probes had the same unique modal action across all three models.
- Relative to the single fixed answer, the unique balanced mode changed on 21 Sol probes, 25 Terra
  probes, and 14 Luna probes; another 32 model/probe results became ties containing the fixed answer.
- Although every canonical action occupied every displayed position once per probe, displayed option
  A was selected in 39.7% of Sol responses, 33.0% of Terra responses, and 32.7% of Luna responses.
  This is an observed position sensitivity in this selected sample, not an estimate for all probes.
- Only three probes produced the same action on all four rotations for all three models. Their actions
  remain candidate evidence, not universal prompt rules. One additional probe was stable in all three
  models but produced a model-specific divergence.

These findings prohibit deriving runtime guidance from a one-shot model preference. The raw selection
is useful for forming a concrete behavioral hypothesis; candidate prompts must still be conditioned on
an explicit user objective and validated on held-out outcome tasks. The selected 78 probes were chosen
because the fixed run diverged, so their instability rate must not be generalized to the other 606.

Authoritative artifacts are under `bench/runs/` with the
`20260804-divergent-counterbalance` prefix. `COMPLETE_RAW_REPORT.md` retains every rotated answer,
`STABILITY_REPORT.md` retains per-model repetitions, `OPTION_ORDER_REPORT.md` compares fixed and
balanced modes, and `EXECUTIVE_ANALYSIS.md` groups all concrete actions by category. Their JSON peers
support reproducible downstream analysis. `MODEL_DOSSIERS.md` summarizes stable behavioral
tendencies, while `MODEL_DECISION_MAPS.md` reconstructs explicit decision maps, tensions, and
uncertainty boundaries without presenting them as hidden chain-of-thought.
`MODEL_CATEGORY_MAPS.md` pivots all 78 probes into model-first, category-second sections and retains
the fixed action, four-position counts, modal policy, evidence strength, and interpretation boundary.
`bench.counterbalanced_analysis` regenerates the executive report, and
`bench.option_order_report --manifest ...` safely selects a frozen subset from a larger fixed-order
artifact.

To turn the raw prior into a prompt-authoring brief for one explicit user objective, compare every
concrete modal action with the profile-scored actions instead of converting a model-wide axis average
into prompt text:

```bash
uv run python -m bench.counterbalanced_prompt_brief \
  counterbalanced.executive-analysis.json probes.jsonl user-profile.json Sol \
  sol-authoring-brief.md sol-authoring-brief.json
```

The brief separates stable behavior already aligned with the profile, stable conflicts worth testing
with small compensating guidance, position-sensitive hypotheses needing replication, and normative
evidence. It never emits deployable guidance. A conflict means only that an unprofiled raw prior chose
a different acceptable trade-off from this user profile; held-out paired evaluation must still prove
that a candidate improves outcomes.

Candidate text has a second fail-closed boundary. Bind a small pool to the exact brief hash, immutable
model identity, explicit utility-profile hash, one role, cited probe IDs, expected effect, and named
regression risks, then compile inert run conditions with:

```bash
uv run python -m bench.prompt_candidate_pool \
  candidate-pool.json profile-conditioned-brief.json compiled-candidates.json
```

Preference candidates must be marked advisory, cannot cite stable aligned behavior as a defect,
cannot use normative evidence as preference evidence, cannot expose numeric utility weights, and are
rejected for absolute language such as “always” or “regardless of the user.” The compiler limits pool
and fragment size, keeps `current` as the no-system-message baseline, derives the concrete target
actions from the cited brief, and always emits `deployment_approved: false`. Compilation produces
evaluation conditions, not a runtime profile.

## Per-model prompt calibration

Treat each prompt fragment as a named, immutable condition. Run the same training probes for
candidate discovery, then generate the report again on held-out validation probes. There are two
selection objectives. `accuracy` optimizes normative behavior. `utility` optimizes for one explicit
user profile while treating normative accuracy, safety, errors, latency, and tool use as hard gates.
Select a profile-specific prompt for an immutable provider/model/revision identity with:

```bash
uv run python -m bench.prompt_calibration validation-report.json \
  --model provider/model@revision --baseline current --objective utility \
  --output model-profile.json
```

For `accuracy`, the selector requires enough paired normative samples and a significant positive
accuracy delta. For `utility`, it requires enough paired preference samples and a significant
positive utility delta, while rejecting normative accuracy regression. Both modes enforce error,
latency, and tool-call ceilings and keep the baseline as a safe fallback. The output binds the
decision to the report and utility-profile hashes. Therefore Infinidev can maintain different
validated prompt profiles for, for example, a high-control user and a high-autonomy user without
pretending either is universally optimal. Profiles must be invalidated on model revision, tool
schema, utility definition, or material harness changes. Calibration is an offline release process;
production conversations must not continuously mutate their own system prompt or infer consequential
preferences from weak evidence.

Candidate generation can follow Automatic Prompt Engineer's pattern: treat an instruction as a
program, ask a model to propose a bounded pool, and score every candidate on the calibration set.
Generation and selection remain separate. The selected candidate must still pass held-out
validation and the deterministic gates above. Published follow-up work reports that optimizer
quality and cost can vary sharply with model scale and meta-instructions, so automatic generation
is optional and the current prompt always remains a candidate.

Compile the selected guidance into an opt-in runtime profile only after the release review:

```bash
uv run python -m bench.prompt_calibration validation-report.json \
  --model provider/model@revision --baseline current \
  --run-config run.json --provider provider --role developer \
  --profile-output model-profile.json --approve-deployment
```

Set `INFINIDEV_PROMPT_CALIBRATION_PROFILE`, `INFINIDEV_PROMPT_CALIBRATION_MODEL_IDENTITY`, and
`INFINIDEV_PROMPT_CALIBRATION_UTILITY_PROFILE` plus its
`INFINIDEV_PROMPT_CALIBRATION_UTILITY_PROFILE_SHA256` to activate it. Infinidev verifies the schema,
deployment approval, exact provider/model route, immutable model identity, active user-profile name
and definition hash, guidance hash and byte count, the 4 KiB guidance ceiling, and role before
appending guidance. Missing, mismatched, oversized, unapproved, or tampered profiles preserve the
existing prompt unchanged. A deployment profile authorizes exactly
one calibrated role; passing developer probes cannot copy the same guidance into the chat agent or
planner. A utility-optimized deployment profile records the utility-profile name and hash used to
select it, while the active name comes from explicit user/project configuration. Production
selection may eventually use sufficiently strong feedback, but must never choose
a preference profile merely because it scores highest for some other user.

## Research-derived priorities

1. **Context placement and retrieval.** Long context capacity is not equivalent to reliable use.
   Liu et al. found a strong position effect, often worst when relevant evidence was in the middle.
   Test ranked context against full context with evidence at several positions.
2. **Reasoning plus action.** ReAct showed gains from interleaving reasoning with external actions
   on question answering and interactive environments. Test tool access and tool descriptions as
   separate conditions; measure success and unnecessary calls.
3. **Reasoning prompts.** Chain-of-thought prompting improved several multi-step benchmarks, but
   the effect is model- and task-dependent. Treat concise reasoning guidance as a condition, not a
   universal rule.
4. **Explanation faithfulness.** Interventions on generated chains show that explanations can be
   weakly connected to answers and that faithfulness varies by task and model. Use controlled
   perturbations and outcomes to characterize behavior; never infer cognition from prose alone.
5. **Agent-level evaluation.** Multiple-choice probes are diagnostic, not sufficient. Maintain a
   second tier of repository tasks with deterministic tests, permission scenarios, recovery from
   tool errors, user-message interruption, and end-to-end completion/cost metrics.
6. **Option-order bias must be observable in the report.** Turpin et al. showed that answer choices
   and few-shot option ordering can bias selections while the generated explanation fails to mention
   that influence. Every family therefore permutes option positions and the qualitative dossier
   reports the selected action text per variant, not merely whether the letter changed.
7. **Optimization consumes examples and feedback, not only scores.** DSPy demonstrates compilation
   of declarative LM pipelines from metrics and collected demonstrations; TextGrad demonstrates
   optimization driven by textual feedback through compound systems. These results motivate an
   evidence-first candidate-generation stage, but do not establish that unconstrained rewriting is
   safe. Infinidev keeps selected actions, expressed criteria, failures, and strengths in the
   authoring brief, then evaluates every proposed fragment against an unchanged held-out baseline.
8. **Elicitation is an experimental factor.** Molfese et al. found a trade-off between format
   constraints and free-form reasoning in MCQ evaluation, while Wang et al. found severe mismatch
   between first-token probabilities and final text answers. The lab therefore preserves final text
   and separates minimal choice elicitation from explanation/confidence elicitation rather than
   treating them as interchangeable measurements.
9. **Verbal confidence is not automatically correctness probability.** Recent controlled evidence
   finds verbal confidence tracks commitment more strongly than correctness. It is retained as a
   behavior-facing self-report under `self_report`, never synthesized for `choice_only`, and never
   used as calibration evidence without identifying the elicitation protocol.

## Explicit runtime user preferences

Preference probes have no universal gold answer, so a selected calibration profile must be bound to
the user's explicit objective rather than inferred silently from one conversation. Runtime profiles
use the same `name`, `description`, and utility-axis weights as offline evaluation, plus
`schema_version=1` and `provenance=explicit_user`. Validate a profile and obtain its canonical hash:

```bash
uv run python -m infinidev.engine.user_preferences examples/user-preference-profile.json
```

Set `USER_PREFERENCE_PROFILE` and `USER_PREFERENCE_PROFILE_SHA256` in `/settings` (or their
`INFINIDEV_` environment equivalents). The model receives only the natural-language description;
numeric utility weights remain evaluation metadata. The rendered block explicitly limits the
profile to trade-offs left open by the current request and cannot override safety, authorization,
repository rules, or evidence.

If a release-gated calibrated prompt profile is active, its validation metadata must match this
same name and canonical hash. A missing, edited, model-inferred, malformed, oversized, or
hash-mismatched preference profile fails closed: no preference block is injected and it cannot
authorize calibrated guidance. Legacy calibration utility settings remain available only when no
explicit runtime profile is selected.

## Artifact-grounded harness feedback

Behavioral choices and self-reported criteria do not answer whether a model encountered avoidable
friction in the harness. `bench.harness_feedback` defines a separate protocol for reviewed visible
artifacts. A response either states that no change is warranted or supplies one concrete friction,
artifact evidence, the smallest suggested change, expected observable effect, regression risk, and
a paired falsifiable experiment. Raw text remains authoritative; the report never ranks a proposal
as truth or activates it.

`bench.run_harness_feedback` sends one user message with no system prompt, uses a fresh isolated
conversation per case, disables LiteLLM retries and caching, enforces a two-second minimum between
request starts, fsyncs each result, and stops on rate limits or transport failures. It shares one
host-wide subscription lock with behavioral probes and repository-task evaluations, so two kinds
of study cannot overlap accidentally.

```bash
uv run python bench/run_harness_feedback.py \
  bench/harness_feedback_cases.example.jsonl \
  bench/harness_feedback_run.openai_subscription.example.json \
  bench/harness_feedback_observations.jsonl \
  --split calibration --repetitions 1 --include-drafts

uv run python -m bench.harness_feedback \
  bench/harness_feedback_cases.example.jsonl \
  bench/harness_feedback_observations.jsonl \
  harness-feedback.md harness-feedback.json
```

The checked-in nine-case file is a draft protocol example, not an approved campaign. Model
agreement may prioritize what to test next, but cannot replace independent review or held-out
outcome evidence. This distinction follows the same explanation-faithfulness boundary used for
MCQ self-reports.

## Held-out context-delivery evaluation

Multiple-choice context preferences do not establish repository-task benefit. The second evaluation
tier uses `bench.context_delivery_eval` with three immutable treatments: no automatic context but
Ken tools available (`baseline`), bounded automatic ContextRank plus the same tools (`ranked`), and
the complete predeclared evidence corpus plus the same tools (`full`). All other model, prompt, tool,
fixture, and generation settings must remain identical.

Tasks are JSONL records with a fixture, user request, deterministic verifier, required evidence, and
an optional front/middle/end placement variant. Placement variants share one family and must remain
in the same calibration or validation split. Every task repetition needs all three conditions from
fresh fixture state. The dataset hash, condition-manifest hash, canonical condition hash, and
immutable model identity are stored on every observation; incomplete pairs and drift are rejected.

The evaluator reports deterministic success, provider errors, prompt/completion tokens, latency,
tool calls, delivered and omitted required evidence, and a path to the raw run artifact. It preserves
every task record before aggregates. Evidence recall is diagnostic only: delivery does not prove the
model used the evidence. Likewise, a passing verifier proves only the fixture's declared checks.

Collection uses Infinidev's actual `LoopEngine` and developer tools. Each treatment gets a new
agent, session, project id, indexed fixture copy, and filesystem copy. `ranked` waits for that
fixture's symbol embeddings; `full` injects the exact UTF-8 corpus through a dedicated user-context
block, never through the system prompt. The collector writes and fsyncs one observation before
moving on, refuses an existing non-empty output, disables client, parse, malformed-call, and
function-calling fallback retries, and stops at the first provider/runtime error (including 429).
It does not retry or resume campaigns implicitly.

```bash
uv run python bench/context_delivery_run.py \
  bench/context_delivery_tasks.example.jsonl \
  bench/context_delivery_conditions.example.json \
  bench/context_delivery_run.openai_subscription.example.json \
  bench/context_delivery_observations.jsonl \
  bench/context_delivery_artifacts \
  --fixture-root bench/context_delivery_fixtures --split validation
```

```bash
uv run python -m bench.context_delivery_eval \
  bench/context_delivery_tasks.jsonl \
  bench/context_delivery_conditions.json \
  bench/context_delivery_observations.jsonl \
  context-delivery.md context-delivery.json --split validation
```

`bench/context_delivery_tasks.example.jsonl` and
`bench/context_delivery_conditions.example.json` contain a reviewed two-family, six-task pilot.
Each fixture is known failing before repair and passing after the narrowly intended repair; this is
still a pilot, not a completed experiment or broad capability claim. Subscription-backed collection
retains the behavioral lab's global single-flight, fresh-session, minimum two-second task-start
pacing, no-retry, and stop-on-429 controls.

## Provider-neutral prompt falsification pilot

`bench.agent_task_run` evaluates a baseline against one model-specific candidate on six held-out
repository tasks: planning, reversible implementation under ambiguity, test selection, code review,
user-owned trade-offs, and recovery from a failed tool. The task schema, evaluator, artifact format,
and runner are provider-neutral. A route file supplies only `provider`, `model`, immutable model
identity, and bounded execution settings, so the same protocol can use any provider supported by
Infinidev, including OpenAI subscription routes, Anthropic, Kimi, and MiniMax.

Each condition receives a fresh copied workspace, agent, and session. Runs are globally
single-flight and ordered task, then baseline/candidate. Every internal LLM request is paced by at
least two seconds, automatic retries are disabled, and the campaign stops on the first provider or
runtime error. Every run preserves its final answer, actions, prompt composition, request payload,
workspace, verifier output, token counts, latency, and tool-call count.

Before provider calls, prove verifier reachability and export a candidate-blind review packet:

```bash
uv run python -m bench.agent_task_preflight \
  bench/agent_task_pilot.tasks.jsonl bench/agent_task_fixtures \
  bench/agent_task_reference_solutions bench/agent_task_pilot.preflight.json

uv run python -m bench.agent_task_review export \
  bench/agent_task_pilot.tasks.jsonl bench/agent_task_fixtures \
  bench/agent_task_pilot.review-packet.json \
  --preflight bench/agent_task_pilot.preflight.json \
  --markdown bench/agent_task_pilot.REVIEW_DOSSIER.md \
  --reviews-template bench/agent_task_pilot.reviews.template.jsonl
```

A documented pre-execution review must approve each deterministic verifier and human rubric. The
review artifact records reviewer identity and whether it was author-side or independent; it must not
claim independence when the task author performs it. Applying that report creates a new approved
JSONL dataset; because its bytes and hash change, regenerate all condition manifests before
collection. Draft tasks cannot run unless a caller deliberately supplies `--include-drafts`, which
is forbidden for this pilot.

The exact regeneration inputs are retained in-tree: the shared explicit utility profile and the
three compiled candidate artifacts. After applying the review report, rebuild manifests with:

```bash
uv run python -m bench.agent_task_manifest bench/agent_task_pilot.approved.jsonl \
  bench/runs/20260804-sol.quality-control.compiled-candidates.json \
  bench/agent_task_pilot.utility-profile.json quality-control-verification \
  bench/agent_task_pilot.sol.approved.conditions.json

uv run python -m bench.agent_task_manifest bench/agent_task_pilot.approved.jsonl \
  bench/runs/20260804-terra.quality-control.compiled-candidates.json \
  bench/agent_task_pilot.utility-profile.json quality-control-explanation-depth \
  bench/agent_task_pilot.terra.approved.conditions.json

uv run python -m bench.agent_task_manifest bench/agent_task_pilot.approved.jsonl \
  bench/runs/20260804-luna.quality-control.compiled-candidates.json \
  bench/agent_task_pilot.utility-profile.json quality-control-decision-ownership \
  bench/agent_task_pilot.luna.approved.conditions.json
```

The three checked-in draft manifests describe 12 executions each, 36 total. Audit count, identity,
isolation, pacing, retry, and stop-on-error invariants without contacting a provider:

```bash
uv run python -m bench.agent_task_campaign_audit \
  bench/agent_task_pilot.tasks.jsonl bench/agent_task_pilot.audit.json \
  --route bench/agent_task_pilot.sol.conditions.json bench/agent_task_run.gpt-5.6-sol.json \
  --route bench/agent_task_pilot.terra.conditions.json bench/agent_task_run.gpt-5.6-terra.json \
  --route bench/agent_task_pilot.luna.conditions.json bench/agent_task_run.gpt-5.6-luna.json
```

Once the reviewed dataset and regenerated manifests pass that audit, use the multi-route command so
all 36 executions share one host-wide lock and one fail-fast boundary:

```bash
uv run python -m bench.agent_task_multi_run \
  bench/agent_task_pilot.approved.jsonl bench/runs/agent-task-pilot \
  --route sol bench/agent_task_pilot.sol.approved.conditions.json bench/agent_task_run.gpt-5.6-sol.json \
  --route terra bench/agent_task_pilot.terra.approved.conditions.json bench/agent_task_run.gpt-5.6-terra.json \
  --route luna bench/agent_task_pilot.luna.approved.conditions.json bench/agent_task_run.gpt-5.6-luna.json
```

The command rejects draft tasks, anything other than three unique model identities, a nonempty
output directory, unsafe execution contracts, or any total other than 36. It automatically writes
per-model observations, complete run artifacts, and deterministic Markdown/JSON reports. It never
resumes a partial campaign implicitly.

For non-OpenAI routes, use the Anthropic, Kimi, or MiniMax example route files. Each route resolves
the provider's registered model prefix and endpoint, reads secrets only from the named environment
variable, forces LiteLLM retries to zero, and restores the user's previous runtime configuration.

This pilot can cheaply falsify a candidate. It cannot establish a model's internal reasoning,
authorize prompt deployment, or replace a larger independently reviewed held-out evaluation.

After collection, first generate the deterministic paired report for each model. Then export the
human evidence into A/B packets. The condition key is written separately and must remain hidden from
the reviewer:

```bash
uv run python -m bench.agent_task_outcome_review export \
  MODEL.agent-task-report.json MODEL.blind-packet.json MODEL.condition-key.json \
  --reviews-template MODEL.blind-reviews.jsonl

uv run python -m bench.agent_task_outcome_review score \
  MODEL.agent-task-report.json MODEL.blind-packet.json MODEL.condition-key.json \
  MODEL.blind-reviews.jsonl MODEL.outcome-decision.json
```

The scorer fails closed on missing judgments or hash drift. It rejects competence, authorization,
preference, single-domain, and material efficiency regressions; recognizes a correct baseline or a
candidate with no effect; and can only advance a multi-domain improvement to a larger calibration
campaign. It always leaves deployment unauthorized.

After all three route reports and blind outcome reviews exist, consolidate the campaign into one
provider-neutral evidence dossier:

```bash
uv run python -m bench.agent_task_campaign_report \
  bench/runs/agent-task-pilot \
  bench/runs/agent-task-pilot/CAMPAIGN_DOSSIER.json \
  bench/runs/agent-task-pilot/MODEL_DECISION_MAPS.md \
  --route sol --route terra --route luna \
  --outcome gpt-5.6-sol bench/runs/agent-task-pilot/sol/outcome-decision.json \
  --outcome gpt-5.6-terra bench/runs/agent-task-pilot/terra/outcome-decision.json \
  --outcome gpt-5.6-luna bench/runs/agent-task-pilot/luna/outcome-decision.json
```

The JSON preserves every final answer, tool trace, action record, verifier result, changed path, and
per-category baseline/candidate comparison. The Markdown is a readable model-by-model decision map.
It deliberately describes observable strategies rather than claiming access to hidden reasoning or
a literal model mental state.

## Prompt responsibility layers

Runtime prompt composition uses four explicit responsibilities from
`infinidev.engine.prompt_layers`: `behavior`, `execution-policy`, `objective`, and
`context-evidence`. System prompts render behavior and execution policy as separate,
provenance-marked layers. Existing user-prompt sections are classified under the same contract.

Active user preferences and model-calibrated preference behavior can enter only the behavior layer.
Deployment profiles use schema version 2 and must declare both `prompt_layer: behavior` and
`evidence_kind: preference_behavior`; older profiles and role entries without behavior-layer
provenance fail closed. This prevents a preference questionnaire from silently compiling into an
operational rule such as always running more tests. Execution-policy calibration needs separate
objective-comprehension and task-execution evidence.

## Isolated prompt-comprehension study

`bench.prompt_comprehension` and `bench.run_prompt_comprehension` measure what a model reconstructs
from requests before execution. Every call is a fresh conversation. The `raw` condition contains
exactly one user message and no system message; comparison conditions can add a typed behavior shell
or the same behavior shell plus an execution policy without replacing the objective or context.

The response contract retains a free reconstruction plus structured objective, deliverables,
constraints, user-owned decisions, authorized and unauthorized actions, verification,
ambiguities, and stop conditions. Reports retain the raw response and all concrete fields. Numeric
summaries describe collection health and cost only; they are not treated as an optimal behavior
score.

The complete authoring battery contains 672 draft cases: 224 controlled three-variant families.
Of these, 432 cases cross 18 linguistic/pragmatic phenomena with eight Infinidev task domains, and
240 cases materialize ten execution-policy comprehension questions in every domain. This is a deliberate
coverage design derived from the Infinidev problem map, not 672 unrelated paraphrases. Every family
records the product problem, research question, useful model information, competing hypotheses,
evidence fields, possible interventions, and held-out confirmation. The 12-case seed catalog remains
only as schema history; campaign input is `bench/prompt_comprehension_battery.draft.jsonl`.

Regenerate and audit the complete draft battery:

```bash
uv run python -m bench.generate_prompt_comprehension_battery \
  bench/prompt_comprehension_battery.draft.jsonl \
  bench/prompt_comprehension_family_registry.json
uv run python -m bench.prompt_comprehension_battery_audit \
  bench/prompt_comprehension_battery.draft.jsonl \
  bench/prompt_comprehension_family_registry.json \
  bench/prompt_comprehension_battery.audit.json
```

The structural audit requires exactly 672 unique requests, complete interpretation keys, atomic
calibration/validation families, balance across domains and phenomena, identical keys for declared
equivalents, changed keys for semantic contrasts, and full problem-to-intervention traceability.
Passing does not self-approve semantic labels: cases remain `draft` until independent content review.

```bash
uv run python -m bench.run_prompt_comprehension \
  APPROVED_CASES.jsonl ROUTE.json OUTPUT.observations.jsonl --split validation

uv run python -m bench.prompt_comprehension_report \
  APPROVED_CASES.jsonl OUTPUT.observations.jsonl \
  OUTPUT.report.json OUTPUT.report.md \
  --registry bench/prompt_comprehension_family_registry.json
```

Future comprehension campaigns default to one-second start pacing, remain globally single-flight,
disable LiteLLM retries and caching, and stop at the first 429. A route may raise the interval;
0.75 seconds is the hard lower configuration bound and is not the default.

Primary references:

- Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*, NeurIPS 2022,
  https://arxiv.org/abs/2201.11903
- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, ICLR 2023,
  https://arxiv.org/abs/2210.03629
- Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, TACL 2024,
  https://arxiv.org/abs/2307.03172
- Lanham et al., *Measuring Faithfulness in Chain-of-Thought Reasoning*, 2023,
  https://arxiv.org/abs/2307.13702
- Turpin et al., *Language Models Don't Always Say What They Think: Unfaithful Explanations in
  Chain-of-Thought Prompting*, 2023, https://arxiv.org/abs/2305.04388
- Liu et al., *AgentBench: Evaluating LLMs as Agents*, ICLR 2024,
  https://arxiv.org/abs/2308.03688
- Zhou et al., *Large Language Models Are Human-Level Prompt Engineers*, ICLR 2023,
  https://arxiv.org/abs/2211.01910
- Yang et al., *Revisiting OPRO: The Limitations of Small-Scale LLMs as Optimizers*, 2024,
  https://arxiv.org/abs/2405.10276
- Khattab et al., *DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines*,
  2023, https://arxiv.org/abs/2310.03714
- Yuksekgonul et al., *TextGrad: Automatic Differentiation via Text*, 2024,
  https://arxiv.org/abs/2406.07496
- Pezeshkpour and Hruschka, *Large Language Models Sensitivity to the Order of Options in
  Multiple-Choice Questions*, Findings of NAACL 2024,
  https://aclanthology.org/2024.findings-naacl.130/
- Wang et al., *My Answer is C: First-Token Probabilities Do Not Match Text Answers in
  Instruction-Tuned Language Models*, Findings of ACL 2024,
  https://aclanthology.org/2024.findings-acl.441/
- Molfese et al., *Right Answer, Wrong Score: Uncovering the Inconsistencies of LLM Evaluation in
  Multiple-Choice Question Answering*, Findings of ACL 2025,
  https://aclanthology.org/2025.findings-acl.950/
- Kumaran, *Reported Confidence in LLMs Tracks Commitment More Than Correctness*, 2026,
  https://arxiv.org/abs/2606.29490

## Rollout gate

A harness change graduates only when its preregistered normative metric or profile-conditioned
utility improves on held-out probes without a material regression in task completion, error rate,
latency/cost, or safety behavior. Preference utility never compensates for violating a hard
normative gate. Record null and negative results. Model upgrades and utility-profile changes rerun
the suite because techniques that help one model or user objective may not help another.
