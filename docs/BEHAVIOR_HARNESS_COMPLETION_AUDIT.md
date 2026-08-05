# Behavior-harness completion audit

Date: 2026-08-04. This audit distinguishes implemented mechanisms, completed empirical evidence, and
remaining release evidence. It does not treat passing unit tests or a protocol document as proof that
an external model study has completed.

## Requirements derived from the project objective

| Requirement | Current evidence | Status |
|---|---|---|
| Understand observable model decisions across many engineering categories | 684 draft probes in 29 authored categories; complete fixed-order run for Sol, Terra, and Luna; 2,052 valid responses | Achieved for exploratory raw behavior |
| Control MCQ option-position effects | Frozen 78-probe divergence manifest; four complete rotations for each model; 936/936 valid responses; option-order and stability reports | Achieved for the selected divergence follow-up |
| Preserve concrete answers rather than only axis scores | Raw JSONL, complete action report, stability report, model dossiers, decision maps, and model/category maps retain selected action text and counts | Achieved |
| Describe how each model decides, separated by category | `bench/runs/20260804-divergent-counterbalance.MODEL_CATEGORY_MAPS.md`: 27 selected categories, 78 unique probes per model | Achieved for the selected divergence follow-up |
| Avoid claiming access to hidden reasoning | Choice-only protocol and every report state the observable-behavior boundary; self-report is specified as a separate experiment | Achieved |
| Adapt prompts to different user objectives rather than seek one universal optimum | Explicit hash-bound user profiles, profile-conditioned utilities, per-model prompt briefs, and fail-closed runtime profile selection | Mechanism achieved; outcome validation pending |
| Generate small, traceable prompt candidates | Candidate pools bind brief hash, model identity, role, user-profile hash, probe IDs, concrete target actions, expected effect, and regression risks | Mechanism achieved |
| Prevent prompt candidates from becoming production rules automatically | Candidate compiler emits `deployment_approved: false`; held-out paired validation and explicit deployment approval remain mandatory | Achieved |
| Ground harness design in research with explicit limitations | `bench/model_behavior_evidence.json` currently audits 20 sources and 14 decisions with zero registry errors | Mechanism achieved; evidence strengths remain mixed by design |
| Listen to model feedback about harness friction | Reviewed feedback schema, isolated runner, reporter, nine draft cases, and safety tests exist | Protocol achieved; no completed real feedback campaign found |
| Prove intelligent context delivery improves repository-task outcomes | Baseline/ranked/full evaluator, fixtures, conditions, artifact capture, and tests exist | Not achieved empirically |
| Avoid prompt overload | Prompt-composition measurement, compact guidance limits, per-role profiles, and candidate byte limits exist | Mechanism achieved; comparative outcome evidence pending |
| Provide strong progress/final feedback to the user | Runtime feedback and final-handoff behavior have focused tests; all models selected the final-handoff policy 4/4 in the normative counterbalanced probe | Achieved as implementation plus diagnostic evidence |
| Independently review the authored probes | Blind family-atomic review/export/apply mechanism exists | Not achieved for the 684-question draft dataset |
| Validate prompt candidates on held-out agent tasks | Calibration selector and deterministic gates exist | Not achieved for the new Sol/Terra/Luna candidates |

## Completed empirical artifacts

- Fixed raw baseline: `bench/runs/20260804-all-684.*.raw.observations.jsonl`.
- Frozen divergence selection:
  `bench/runs/20260804-all-684.divergent-counterbalance.manifest.json`.
- Counterbalanced raw observations:
  `bench/runs/20260804-divergent-counterbalance.gpt-5.6-*.observations.jsonl`.
- Complete response comparison:
  `bench/runs/20260804-divergent-counterbalance.COMPLETE_RAW_REPORT.md`.
- Stability and option-order evidence:
  `STABILITY_REPORT.md` and `OPTION_ORDER_REPORT.md` under the same prefix.
- Human-readable decision maps:
  `MODEL_DECISION_MAPS.md`, `MODEL_DOSSIERS.md`, and `MODEL_CATEGORY_MAPS.md`.
- Machine-readable analysis and profile-conditioned briefs: JSON peers under the same prefix.
- Inert candidate conditions for the quality-and-control profile:
  `bench/runs/20260804-{sol,terra,luna}.quality-control.compiled-candidates.json`.

## Evidence that contradicts full completion

The two existing Sol context-delivery observation files contain one row each, both for
`rollback-front` under the baseline condition. The v2 row records `success: false` and a DNS
resolution error before model completion. There is no complete baseline/ranked/full comparison, so
the current worktree cannot support a claim that automatic ContextRank improves task outcomes.

No real harness-feedback observation or report artifact was found under `bench/runs/`. The feedback
protocol is implemented and tested, but no Sol/Terra/Luna feedback conclusions are currently
available.

All 684 behavior probes remain drafts. The three normative divergences became compatible with their
draft keys after counterbalancing, but that does not replace independent blind review of every family.

The three quality-and-control candidate fragments are compiled evaluation conditions only. They have
not been run on calibration and held-out validation tasks, so no runtime calibrated profile should be
approved from them.

## Next evidence gates

1. Independently review and version an approved subset before using normative accuracy for release.
2. Freeze profile-conditioned calibration and validation manifests for each role/model candidate.
3. Run candidate versus unchanged baseline with the explicit quality-and-control profile; require
   paired utility improvement, no normative regression, and error/latency/tool-call gates.
4. Complete baseline/ranked/full context-delivery tasks across placement families and all selected
   models; inspect generated artifacts and deterministic verifier output.
5. Run the reviewed harness-feedback campaign separately, preserve raw text, and turn suggestions only
   into falsifiable paired experiments.
6. Measure total system/user/tool-schema prompt composition for baseline and any winning candidate so
   a local gain cannot silently create product-wide prompt pollution.

Provider calls for these new campaigns should not begin implicitly. Freeze their manifests and obtain
operator approval for the exact call count and route, then preserve the existing global single-flight,
two-second interval, isolated-conversation, no-retry, and stop-on-first-rate-limit contract.
