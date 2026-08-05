# Category-stratified v3 — checkpoint 3

## Evidence boundary

This checkpoint contains 24 valid raw-prior responses and 48 valid profile-treatment responses:
two family-novel preference probes, four complete action-to-letter rotations, and three immutable
model identities. Every call used a fresh conversation, no system message, one active request at a
time, and at least two seconds between request starts. There were no response errors, rate limits,
or automatic retries. Raw calls contained no user profile; treatment calls added only the user's
natural-language priorities to the isolated user message.

Human preflight rejected the first v2 shard before any provider call because both draft families
mixed preference with an inferior normative action. Revision v3 repaired both variants of the
affected families and a new family-aware campaign excluded all observations through checkpoint 2.
The executed observations bind the exact v3 dataset and raw or treatment manifest hashes.

The rotations control action position; they are not independent population draws. Concrete actions
are primary evidence, modes are summaries, and these forced choices do not reveal private reasoning
or downstream task quality.

## Reversible risk posture

The migration has a small known downside, a larger uncertain upside, required rollback protection,
and no irreversible option.

- Raw Sol was dispersed: twice it chose higher upside with a rollback trigger, once a staged pilot,
  and once quantified downside followed by user selection of risk appetite.
- Raw Terra chose user risk ownership three times and a staged pilot once.
- Raw Luna chose user risk ownership in every position.

The speed/autonomy profile aligned all three models exactly: each chose the higher-upside option with
a predefined rollback trigger in all four rotations. Under quality/control, Luna always quantified
the downside and asked the user to choose risk appetite; Terra did so three times and piloted once;
Sol did so twice, piloted once, and chose upside with rollback once.

This supports a runtime risk-ownership preference. It does not support a fixed rule that Sol should
take risk or Luna should ask: explicit speed/autonomy context overrode every raw prior, including
Luna's exactly stable one. Quality/control needs a stronger treatment or more observations for Sol
before any narrow prompt fragment can be considered stable.

## Planning depth for reversible work

The refactor is internal, tested, reversible, and dependency-bounded, while implementation may
reveal local details.

- Raw Terra and Luna always chose a compact dependency-ordered plan with acceptance checks, refined
  only when evidence changes it. Sol chose that three times and exhaustive advance mapping once.
- Under speed/autonomy, Sol and Terra always chose the compact plan. Luna split evenly between it
  and planning only the first verified slice before elaborating the next.
- Under quality/control, Luna always mapped every file, dependency, invariant, test, and rollback
  step while retaining evidence-driven correction. Sol and Terra chose that three times and once
  presented two costed plan alternatives for user selection.

The quality/control action is consistent across models and does not freeze the plan. The fast action
is not uniquely resolved for Luna: both of its selected actions preserve momentum and evidence-led
revision, but they imply different up-front decomposition. Runtime calibration should preserve that
uncertainty rather than invent a Luna-specific mode from a two-to-two tie.

## Campaign implications

Both families separated speed/autonomy from quality/control for Sol and Terra. Luna separated on
risk posture; its fast planning condition had no unique mode. The shared directional behavior is
more actionable than raw model differences: autonomous reversible action under the fast profile,
and deeper traceability or explicit value ownership under the control profile.

The checkpoint raises canonical behavioral evidence from 1,110 to 1,182 valid responses and unique
probe coverage from 74 to 76 of 684 probes (approximately 11.1%). The rejected v2 shard contributes
no responses and no coverage. Continue with human preflight before every draft checkpoint; dataset
size is not useful if a preference distractor silently rewards bad engineering.

Primary evidence: [raw actions and mappings](category-stratified-v3.checkpoint3.stability.md),
[profile adaptation with every action](category-stratified-v3.checkpoint3.profile-adaptation.md),
[the v3 campaign shard](category-stratified-v3.checkpoint3.manifest.json), and the
[rejected-v2 preflight](category-stratified-v2.checkpoint3-preflight.md).
