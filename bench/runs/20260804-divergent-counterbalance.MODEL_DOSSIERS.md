# GPT-5.6 Sol, Terra, and Luna: counterbalanced behavioral dossiers

This report characterizes externally observable choices, not hidden chain-of-thought. It uses only
the 78 probes selected because Sol, Terra, and Luna diverged in the 684-question fixed-order run.
Every probe was then asked four times in isolated conversations, with each action occupying every
displayed position once. The resulting tendencies are hypotheses for prompt experiments, not model
personality claims or universal policies.

## Cross-model result

The main result is instability, not a winner. Sol repeated one canonical action in all four rotations
on 24/78 probes, Terra on 26/78, and Luna on 29/78. Only 23 probes had the same unique modal action
across the models, and only three were exactly stable and shared by all three. The complete action
record is in `20260804-divergent-counterbalance.EXECUTIVE_ANALYSIS.md`; the summaries below emphasize
the exactly stable cases so a single position-sensitive answer does not become a story about a model.

All three selected the displayed first option more often than the 25% produced by a stable canonical
choice over a complete four-position cycle: Sol 39.7%, Terra 33.0%, Luna 32.7%. This selected sample
therefore shows strong option-position sensitivity, especially for Sol. It does not estimate the
effect over the other 606 probes.

## Sol

Sol's 24 exactly stable results most often express a pragmatic, reversible, completion-oriented
policy. Concrete examples include inferring a private helper name from the nearest analogue and
disclosing the assumption; deploying atomically with a tested rollback; switching directly to text
search after a tool failure; reusing anchored verified context; accepting focused plus impacted tests;
and presenting the recommendation before evidence and alternatives.

Its stable evidence policy is economical rather than maximal: accept a three-layer proof for exactly
what it establishes, or provide an independent check as an optional user command. Its stable editing
choice permits adjacent cleanup inside the same tested private function, while its stable rollout and
uncertainty choices favor reversible action with telemetry. These actions suggest candidate guidance
for avoiding unnecessary work when a high-autonomy or speed-sensitive user wants it.

This is not simply “autonomous.” Sol consistently asked the user to choose risk posture for an
uncertain pilot and workspace isolation cost, chose one-at-a-time clarification for dependent
questions, and required confirmation before a recoverable 40-file operation. The better hypothesis is
that Sol distinguishes cheap local inference from user-owned risk or scope decisions.

The strongest caution is position sensitivity: Sol had the fewest exactly stable probes and the
largest displayed-A share. A one-shot Sol answer is particularly weak evidence for model-specific
prompt authoring.

## Terra

Terra's 26 exactly stable results combine concise structured communication with explicit user control
over genuine trade-offs. It batched four independent clarification questions with recommended
defaults; consistently led review reports with blockers while collapsing style notes; showed artifact
formats and maintenance cost before asking the user to choose; and asked for scope on optional nearby
cleanup. This suggests candidate guidance for users who want visible decision points without many
serial interruptions.

Terra was also stably autonomous once boundaries were clear. It chose to execute approved reversible
plan steps until evidence, irreversibility, or authorization changed; update all necessary scoped
boundaries without blocking checkpoints; and resume interrupted work by revalidating and reusing prior
evidence. Its communication choices were consistently compact: outcome plus evidence and one
implication, one highest-value optional follow-up, and progress heartbeats during long waits.

On validation, Terra stably favored the directly invalidated unit tests plus the traced integration
case. On consequential choices it requested broader ownership: all affected stakeholders before a
recommendation, and explicit risk posture for a pilot. The evidence supports a “structured autonomy”
hypothesis more than either maximum control or maximum independence.

Terra's balanced unique mode differed from its original fixed answer on 25 probes, the most of the
three. Even though its displayed-A skew was lower than Sol's, fixed-order observations alone would
have mischaracterized many concrete policies.

## Luna

Luna had the most exactly stable probes, 29/78. Its stable decision-support choices were frequently
recommendation-led but expandable: recommend the leader and decisive trade-off, provide a one-line
recommendation with optional deeper analysis, and pair a recommendation with a compact verification
matrix. It also consistently recommended the profile-aligned option on a genuine Pareto frontier and
invited correction.

At evidence boundaries Luna more often preserved user choice. It explained the current evidence limit
and offered deeper research; offered repository history as a 25-minute follow-up; compared fallback
cost and fidelity before switching tools; and presented atomic versus staged rollout failure bounds
for selection. This supports a hypothesis of decisive presentation coupled with explicit escalation
when additional evidence or operational policy has material cost.

Its stable web behavior was concise and bounded: stop after convergent primary sources, cite them, and
state the evidence scope, with history available as an optional expansion. In implementation it kept a
small isolated component behind a common interface rather than creating a premature abstraction. In
long-running interaction it favored concise progress heartbeats and continuing unless risk or plan
divergence increased.

Luna changed unique mode least often relative to the fixed run, 14 probes, but remained unstable on
49/78. “Most stable of the three” still means that most selected divergences did not repeat exactly.

## Shared robust actions

Only three probes produced the same action in all four rotations for every model:

1. For a reversible queue pilot with uncertain upside, quantify pilot cost, upside range, and rollback
   bounds, then ask the user to choose risk posture. This is a preference action and should remain
   conditioned on the user's desired control and evidence cost.
2. When the final response is the durable handoff, state the outcome, important artifacts or cause,
   verification, limitations, and an actionable next step in the final response itself. This is the
   one stable normative result and supports preserving Infinidev's existing final-handoff guidance.
3. When uncertainty does not affect results, proceed reversibly and expand only if it does. This is a
   preference action compatible with low-interruption users, not a rule for consequential ambiguity.

One probe was stable in all models but divergent: for bounded requirement traceability, Terra chose to
show matrix/checklist examples and let the user select, while Sol and Luna chose a compact grouped
checklist linking requirements to implementation and evidence. This is precisely the kind of stable
model-specific prior that can inform a per-model candidate, but the user's desired formality remains
the optimization target.

## Prompt-design decision

No runtime prompt profile is deployed from these answers. The experiment supports four harness
changes instead: require balanced repetitions for MCQ discovery, retain the action text and every
selection, bind preference optimization to an explicit user profile, and treat stable actions only as
candidate-generation evidence. A candidate still needs paired held-out outcome evaluation against the
unchanged prompt before release.
