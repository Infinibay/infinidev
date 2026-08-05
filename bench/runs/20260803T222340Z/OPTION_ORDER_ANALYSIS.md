# Fixed-order versus counterbalanced option analysis

## Why this experiment was necessary

The first 678 responses always attached the same action to the same A/B/C/D label. That design
cannot distinguish preference for an action from sensitivity to its displayed letter or position.
[Zheng et al. (2023)](https://arxiv.org/abs/2309.03882) found systematic option-ID selection bias
across 20 language models and three MCQ benchmarks, so fixed ordering is not a harmless default.

The updated runner now uses a deterministic balanced rotation. For every four-choice question and
model, four isolated repetitions place each canonical action exactly once under A, B, C, and D. The
observation stores the provider's displayed letter, the complete displayed-to-canonical mapping,
and the resulting canonical action separately. The runner rejects incomplete rotation cycles.

This experiment reran the eight predeclared profile-target questions under that protocol for Sol,
Terra, and Luna: 96/96 valid provider responses, zero errors, zero rate limits, no system message,
fresh conversation per call, single-flight execution, and a minimum two-second start interval.

## Result

Fixed and counterbalanced canonical modes had the following relations:

- **Sol:** 6/8 remained the same unique mode; 2/8 changed to a different unique mode.
- **Terra:** 3/8 remained the same unique mode; 4/8 became ties containing the old mode; one test
  family became a tie that excluded the old mode.
- **Luna:** 4/8 remained the same unique mode; 3/8 became ties containing the old mode; one family
  changed to a different unique mode.

Exact four-run counterbalanced action stability was only 4/8 for Sol, 2/8 for Terra, and 3/8 for
Luna. By comparison, the fixed-order three-run data had made several of these families look much
more stable. The reduction cannot be attributed solely to letter bias because the runs also add a
fourth sample and occur later in time, but it is sufficient to invalidate fixed-order stability as a
prompt-calibration gate.

Displayed provider-letter totals were:

- Sol: A=10, B=8, C=6, D=8.
- Terra: A=13, B=7, C=5, D=7.
- Luna: A=10, B=9, C=8, D=5.

With only 32 responses per model these counts are descriptive, not evidence of a statistically
established global letter bias. The action-level changes are the more important finding.

## Concrete corrections

- Sol's fixed test mode accepted focused plus impacted tests and stopped. Counterbalancing instead
  produced a unique mode that delivered provisionally and ran the full suite as a non-blocking
  follow-up.
- Sol's fixed uncertainty mode used an expandable risk analysis. Counterbalancing moved to a concise
  qualifier plus tested fallback.
- Terra's fixed test mode accepted focused plus impacted tests. In the balanced run that action was
  never selected: dependency-selected cross-subsystem testing and provisional/full-suite follow-up
  tied 2–2.
- Luna's fixed independent-verification mode used a risk-weighted second parser. Counterbalancing
  produced a unique mode that treated hash plus deterministic regeneration as sufficient proof.
- Terra and Luna retained several old modes only as one side of a tie. Those cases must be represented
  as unresolved, not as stable model traits.

The complete [option-order report](profile-target-v1.option-order-report.md) preserves every
provider letter, mapping, canonical action, count, and raw reply. The
[counterbalanced stability report](profile-target-v1.counterbalanced-stability.md) provides the
cross-model view.

## Consequences for the laboratory

1. All new four-choice MCQ studies use complete balanced rotations by default.
2. Provider letters are never compared directly across permutations; reports compare canonical
   actions while retaining the presented mapping.
3. Fixed-order historical results remain raw evidence of their exact presentation, but cannot gate
   production prompt calibration.
4. Profile adaptation observed under fixed ordering is a hypothesis until replicated with balanced
   rotations.
5. More repetitions must be whole multiples of four so every action receives equal exposure at
   every letter.

The next profile experiment should therefore counterbalance both user profiles. That is more useful
than expanding fixed-order coverage, because it tests whether the strong 8/8 Sol/Terra adaptation
survives removal of this confound.
