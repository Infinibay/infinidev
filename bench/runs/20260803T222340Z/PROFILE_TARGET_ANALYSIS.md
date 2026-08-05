# Targeted user-profile adaptation analysis

> **Superseded evidence boundary:** this experiment used fixed action-to-letter order. The later
> [counterbalanced study](OPTION_ORDER_ANALYSIS.md) showed that several raw modes changed or became
> tied when actions rotated through A/B/C/D. The profile adaptations below remain hypotheses until
> both profiles are rerun with balanced rotations; they must not gate production prompt calibration.

## Evidence boundary

This experiment predeclared eight preference families after the three-run raw-prior study, then
froze them in [a dataset-bound manifest](profile-target-v1.manifest.json) before running either user
profile. The set contains the one fully stable raw cross-model divergence plus seven stable,
partially stable, or unstable differences selected for their calibration relevance.

For each model and question, three isolated choice-only calls used the speed/autonomy profile and
three used the quality/control profile. Every call had a fresh conversation, no system message, no
question history, a minimum two-second start interval, and no automatic retries. The experiment
produced 144/144 valid provider responses, with zero errors and zero rate limits.

Preference choices have no universal correct answer. The purpose is to observe whether the model
changes its concrete decision when the user's requested outcome changes. A profile has a different
mode only when both three-run conditions have a unique mode; ties remain explicitly unresolved.

## Main result

Sol and Terra had different unique modal actions between the two profiles on all eight targeted
questions. Luna separated on five; two Luna quality/control families produced three-way ties, and
Luna retained the same verification mode across profiles. This is not a quality ranking. It shows
that explicit user priorities can dominate many raw model defaults, while the degree and stability
of adaptation remain model- and family-specific.

The strongest shared adaptations were concrete and directionally appropriate:

- **Implementation boundary:** all three implemented the isolated component behind the common
  interface under speed/autonomy; all three prototyped both boundaries and measured complexity
  under quality/control.
- **Tool failure:** all three immediately switched to the equivalent direct search channel under
  speed/autonomy; all three exposed latency and evidence-equivalence trade-offs before asking
  whether to fall back under quality/control.
- **Vague workflow:** all three built a throwaway prototype under speed/autonomy; all three asked a
  focused workflow question first under quality/control.
- **Testing:** Sol and Terra accepted focused plus impacted tests under speed/autonomy. Luna sampled
  each of three stopping policies once, so its speed-profile test policy is unresolved. Under
  quality/control, all three ran the complete suite and investigated failures before completion.
- **Interruption:** all three finished the safe atomic step before switching under speed/autonomy.
  Under quality/control, Terra and Luna paused at the safe temporary-file boundary; Sol's mode was
  to expose the boundary and eight-second cost and let the user choose, though that Sol condition
  was not exactly stable.
- **Uncertainty:** all three used a concise confidence qualifier plus tested fallback under
  speed/autonomy. Under quality/control Sol and Terra offered a compact/full risk presentation
  choice; Luna selected three distinct richer presentations, so its exact quality-profile artifact
  remains unresolved even though all were more detailed.

These convergences matter: some differences previously attributed to a model disappear when the
user's preference is explicit. The harness should therefore inject the relevant user objective
before adding model-specific behavioral guidance.

## Differences that remain useful

Two families retained meaningful model-specific behavior:

- **Complex multi-slice requirements:** under speed/autonomy, Sol defined contracts for all slices
  and implemented incrementally, while Terra and Luna fully implemented slice one and learned from
  feedback. Under quality/control, Sol and Terra resolved every slice before coding; Luna instead
  drafted all requirements provisionally, implemented slice one, and asked the user to review the
  rest against concrete evidence. The raw Sol/Terra-versus-Luna divergence therefore changed shape
  but did not disappear.
- **Independent verification:** under quality/control, Sol required full independent-parser
  agreement before completion. Terra used a deterministic risk-weighted independent sample. Luna's
  modal action was the same risk-weighted sample under raw, speed, and quality profiles, and the
  quality condition was unstable. Luna may need more explicit audit-depth information than the
  broad quality/control profile provides.

The complete [profile adaptation report](profile-target-v1.adaptation-report.md) lists every offered
action, three-run count, mode, exact-stability flag, and repetition-level selection. Its JSON form
retains the raw provider replies. The individual observation JSONL files remain the authoritative
event record.

## Prompt-calibration implication

The data argues for this order of operations:

1. infer or ask for the user's relevant objective only when the decision family requires it;
2. express that objective compactly in natural language;
3. preserve the model's resulting behavior if repeated actions already satisfy the objective;
4. add small model/family-specific guidance only for a replicated residual mismatch;
5. retest normative safety and nearby preference families before activating the guidance.

A permanent “Sol is autonomous” or “Terra asks first” prompt would be misleading. Both models moved
across all eight targeted families when the user profile changed. Likewise, Luna's lower separation
count does not mean lower responsiveness: some tied quality-profile answers were all qualitatively
more cautious, but the broad profile did not determine one exact presentation.

## Next evidence gate

The next targeted experiment should refine audit depth and artifact format as separate user
preferences, rather than expanding a generic quality scalar. The unresolved Luna verification and
uncertainty families are good candidates. Any production prompt fragment still requires held-out
normative regression testing and independent review of the draft probe families.
