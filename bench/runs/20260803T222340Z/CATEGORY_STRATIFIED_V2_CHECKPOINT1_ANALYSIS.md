# Category-stratified v2 — checkpoint 1

## Evidence boundary

This checkpoint contains 24 valid raw-prior responses: two previously unseen preference families,
four complete action-to-letter rotations, and three model identities. Every call used a fresh
conversation, no system message, no user profile, choice-only elicitation, one active request at a
time, and at least two seconds between request starts. There were no response errors, rate limits,
or automatic retries.

The runner was strengthened immediately after collection to place dataset and manifest hashes in
every observation and in the resume key. This checkpoint's original append-only artifacts remain
preserved; hash-bound copies and per-model source-to-bound lineage files were produced without
changing answers or raw responses. The stability report was regenerated from those bound copies.

The four repetitions are position controls, not four independent draws from an unconstrained
population. Exact stability across the rotations is strong evidence that the selected action did
not depend on its displayed letter in this checkpoint. Instability can reflect position effects,
near-indifference, or sampling variation; four observations cannot identify which mechanism alone.
Because no user preference was supplied, these are model priors rather than recommended universal
policies or measures of quality.

## Code-review reporting

The scenario has two correctness blockers, five maintainability concerns, and twelve optional style
observations, all evidence-linked.

- Sol selected the same action in all four presentations: lead with the two blockers, summarize the
  five concerns, and collapse style notes into an optional section.
- Terra chose that same hierarchy in three presentations and once chose a severity-and-effort table
  with expandable evidence.
- Luna chose the expandable severity-and-effort table in three presentations and the hierarchical
  summary once.

This is a concrete reporting-style difference, not a difference in whether blockers should be
reported. Sol's prior is concise severity-first prose; Luna's observed prior exposes a structured
decision interface; Terra is closer to Sol but not position-stable. A future profile treatment can
test whether concise-reading versus audit/control preferences explain the difference. Until then,
no model-specific system guidance is justified.

## Ownership of a complex decision

Three architectures meet every hard constraint; the remaining cost, latency, and maintainability
weights belong to the user.

- Terra selected the same action under every presentation: show the Pareto frontier without
  choosing and ask the user to set the decisive weight.
- Luna selected the same different action under every presentation: recommend the option aligned
  with the stated profile, explain two decisive trade-offs, and invite correction.
- Sol split two-to-two between those actions, leaving no unique mode.

This is the strongest actionable difference in the checkpoint. Terra defaults to transferring the
unresolved value judgment back to the user; Luna defaults to a recommendation with an explicit
correction path; Sol has no stable raw prior between them. Neither action is universally optimal.
The right calibration target depends on whether a user wants direct control over the final weight or
prefers the agent to convert an already stated profile into a recommendation. The next experiment
should apply opposing control/autonomy profiles to this same family, with complete rotations, before
writing any prompt fragment.

## Campaign implications

The family-level exclusion rule prevented the campaign from counting sibling variants as novel
conceptual coverage. Of 29 categories, 26 still contain an unobserved preference family; ambiguity
and clarification, long-context position, and vague-requirement analysis currently have none. The
checkpoint increases behavioral coverage from 70 to 72 unique probes while adding two new families.
Its 24 responses increase the canonical behavioral-response total from 966 to 990.

Continue in whole-probe checkpoints only. With four-choice probes, each added question costs four
serial calls per model. `--max-probes` preserves the full cycle; `--max-runs` is rejected under
balanced rotation because truncating shuffled calls can produce a falsely balanced artifact.

Primary evidence: [all actions, raw replies, provider letters, and mappings](category-stratified-v2.checkpoint1.stability.md)
and the [frozen 26-family campaign manifest](category-stratified-preference-v2.manifest.json).

## Opposing-profile treatment

The two families were then predeclared for speed/autonomy versus quality/control treatment. Each
condition retained all four action positions for every model, adding 48 valid responses with no
errors or rate limits. These profile-conditioned answers are user-context treatments, not system
prompts.

Decision ownership adapted in the same direction across all three models:

- Under speed/autonomy, Sol chose a reversible default with measured review in all four rotations;
  Terra and Luna chose it in three and recommended-with-correction once.
- Under quality/control, Sol and Terra always presented the Pareto frontier and asked the user to
  set the decisive weight; Luna did so three times and recommended-with-correction once.

The raw Sol tie and the raw Terra/Luna difference therefore do not justify permanent model-specific
rules. Once the user's desired ownership is stated, all three move toward reversible autonomous
action for the fast profile and explicit user weight-setting for the control profile. This is direct
evidence for runtime user preference context over a larger fixed system prompt.

Code-review reporting also adapted, but less uniformly:

- Sol and Terra always used blocker-first concise reporting under speed/autonomy. Under
  quality/control, both selected an interactive short-batch walkthrough three times and a
  severity/effort table once.
- Luna's speed/autonomy condition was unstable: two interactive walkthroughs, one table, and one
  blocker-first summary. Its quality/control condition matched the Sol/Terra 3-to-1 pattern.

For Sol and Terra, concise blocker-first versus interactive category walkthrough is a useful
profile-sensitive reporting policy. Luna did not establish a distinct fast-profile mode robustly
enough to support a model-specific calibration fragment. All concrete repetitions remain available
in the report rather than being replaced by the modal summary.

Primary evidence: [profile adaptation with every action](category-stratified-v2.checkpoint1.profile-adaptation.md)
and the [predeclared treatment manifest](category-stratified-v2.checkpoint1-profile-treatment.manifest.json).
