# Counterbalanced user-profile adaptation analysis

## Evidence boundary

This experiment repeats the eight predeclared preference families under speed/autonomy and
quality/control while rotating every canonical action exactly once through A, B, C, and D. Raw and
both profile conditions therefore share the same complete four-rotation protocol.

The six profile blocks produced 192/192 valid responses. Every call used a fresh conversation, no
system message, choice-only elicitation, single-flight execution, a minimum two-second start
interval, and no automatic retries. There were zero errors and zero rate limits.

Different unique modes indicate observable profile separation, not correctness or model quality.
Ties remain unresolved. Exact stability means all four rotations selected the same canonical action;
a unique 3–1 mode is informative but weaker.

## Corrected adaptation result

- **Sol:** fast/autonomy and quality/control had different unique modes in 7/8 families, down from
  8/8 under fixed order. Exact stability was 3/8 for fast/autonomy and 5/8 for quality/control.
- **Terra:** profile separation was 6/8, down from 8/8. Exact stability was 5/8 and 4/8.
- **Luna:** profile separation remained 5/8, but some specific modes changed. Exact stability was
  3/8 and 5/8.

User preferences therefore remain a strong behavioral treatment after removing balanced
action-position exposure, but fixed ordering exaggerated its breadth for Sol and Terra. A mode change
alone should not be called a stable policy when one or both conditions vary across rotations.

## Adaptations shared by all three models

Two families were exceptionally clean:

- **Implementation boundary:** all three were exactly stable on implementing the isolated component
  behind the common interface under speed/autonomy, and exactly stable on prototyping both
  boundaries with contract-test measurements under quality/control.
- **Vague workflow:** all three were exactly stable on building a throwaway prototype under
  speed/autonomy, and exactly stable on asking a focused workflow question first under
  quality/control.

Other shared directional patterns were real but less stable:

- Under quality/control, all three had the same unique and exactly stable full-suite test action.
  Fast/autonomy favored stopping earlier, but Sol and Terra were not exactly stable and Luna tied
  between a cross-subsystem subset and provisional delivery plus later full-suite evidence.
- All three had a concise qualifier plus tested fallback as the unique fast/autonomy uncertainty
  mode. Under quality/control Sol and Terra preferred offering a compact/full risk presentation
  choice, while Luna preferred the structured likelihood-impact-fallback table.
- Quality/control moved all three to requiring full independent-parser agreement as the unique
  verification mode. This corrects the fixed-order report, where Terra and Luna appeared to prefer
  only a risk-weighted independent sample.

## Residual model and family differences

- Sol's quality/control requirement-iteration answers formed a four-way tie—one selection for every
  action. No requirement policy can be inferred there despite the strong profile text.
- Terra's quality/control interruption policy tied between pausing immediately and exposing the
  eight-second remainder for user choice. Its fast mode finished the safe atomic step.
- Luna's fast tool-fallback policy tied between switching immediately and completing with direct
  evidence before comparing semantic results later. Quality/control stably exposed trade-offs and
  asked whether to fall back.
- Multi-slice requirements remained model-dependent: Terra's balanced fast mode defined contracts
  across slices while Luna implemented one slice first; under quality/control Terra fully resolved
  all decisions, while Luna used incremental contracts rather than the provisional all-slice draft
  seen in the fixed-order run.

## Fixed-order corrections

Comparing each profile to its own fixed-order predecessor showed:

- Fast/autonomy preserved the same unique mode on 7/8 Sol, 6/8 Terra, and 4/8 Luna families.
- Quality/control preserved it on 7/8 Sol, 6/8 Terra, and 4/8 Luna families.
- Notable changes included Sol fast verification, Terra fast requirement iteration, Terra quality
  verification, and Luna quality interruption, requirements, and verification.

The [counterbalanced adaptation report](profile-target-v1.counterbalanced-adaptation.md) contains
every repetition, canonical action, mode, and stability flag. Separate
[fast-profile](profile-target-v1.fast-autonomy.option-order.md) and
[quality-profile](profile-target-v1.quality-control.option-order.md) reports preserve every
fixed-versus-balanced provider letter and mapping.

## Prompt-calibration implication

The result strengthens a narrow claim and weakens a broad one:

- Stronger: explicit user priorities can reliably change several concrete decision families even
  after complete option counterbalancing.
- Weaker: a broad profile does not define a complete model policy, and a unique mode is not always
  exactly stable across equivalent presentations.

Infinidev should therefore condition only the relevant decision family, avoid global personality
labels, and retain a fallback for unresolved or tied behavior. No production prompt fragment should
be generated from a family unless its balanced profile effect is replicated and its normative
neighbors pass held-out regression gates.
