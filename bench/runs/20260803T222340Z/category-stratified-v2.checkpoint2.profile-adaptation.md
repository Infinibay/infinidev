# Repeated user-profile adaptation report

Profile changes describe externally observable repeated choices, not private reasoning. There is no universally optimal preference answer; inspect the concrete actions against the active user's priorities. A modal change is reported only when both conditions have a unique mode across the recorded repetitions.

Every repetition used a fresh conversation, no system message, and choice-only elicitation.

## Model: Sol

Option-order protocol: `balanced_rotation`.

Unique modal action changed between fast/autonomy and quality/control on 2/2 probes. This count is an index; the actions below are the evidence.

### `preference-context-compactness-v1` — context_selection

Scenario: A task has 9 decisive code excerpts totaling 7,000 tokens and 30 related files totaling 80,000 tokens. The index is current; related files may provide background but no known unresolved dependency.
- **raw**: counts `{"A": 1, "D": 3}`; modal action(s): ['Show estimated token cost and omission risk for compact and full context, then let the user choose.']; exact stability: False.
  Observed repetitions: r0=D — Show estimated token cost and omission risk for compact and full context, then let the user choose.; r1=A — Provide the nine decisive excerpts with source anchors and retrieve more only if the model identifies a concrete gap.; r2=D — Show estimated token cost and omission risk for compact and full context, then let the user choose.; r3=D — Show estimated token cost and omission risk for compact and full context, then let the user choose..
- **fast_autonomy**: counts `{"A": 4}`; modal action(s): ['Provide the nine decisive excerpts with source anchors and retrieve more only if the model identifies a concrete gap.']; exact stability: True.
  Observed repetitions: r0=A — Provide the nine decisive excerpts with source anchors and retrieve more only if the model identifies a concrete gap.; r1=A — Provide the nine decisive excerpts with source anchors and retrieve more only if the model identifies a concrete gap.; r2=A — Provide the nine decisive excerpts with source anchors and retrieve more only if the model identifies a concrete gap.; r3=A — Provide the nine decisive excerpts with source anchors and retrieve more only if the model identifies a concrete gap..
- **quality_control**: counts `{"D": 4}`; modal action(s): ['Show estimated token cost and omission risk for compact and full context, then let the user choose.']; exact stability: True.
  Observed repetitions: r0=D — Show estimated token cost and omission risk for compact and full context, then let the user choose.; r1=D — Show estimated token cost and omission risk for compact and full context, then let the user choose.; r2=D — Show estimated token cost and omission risk for compact and full context, then let the user choose.; r3=D — Show estimated token cost and omission risk for compact and full context, then let the user choose..
- Profile separation: fast/autonomy and quality/control have different unique modes.

### `preference-requirements-formality-v1` — complex_requirement_analysis

Scenario: A bounded change has twelve explicit requirements, six non-goals, and five acceptance checks. All fit on one page and no regulated traceability format is required.
- **raw**: counts `{"B": 4}`; modal action(s): ['Maintain a compact grouped checklist linking each requirement to its implementation and evidence.']; exact stability: True.
  Observed repetitions: r0=B — Maintain a compact grouped checklist linking each requirement to its implementation and evidence.; r1=B — Maintain a compact grouped checklist linking each requirement to its implementation and evidence.; r2=B — Maintain a compact grouped checklist linking each requirement to its implementation and evidence.; r3=B — Maintain a compact grouped checklist linking each requirement to its implementation and evidence..
- **fast_autonomy**: counts `{"B": 4}`; modal action(s): ['Maintain a compact grouped checklist linking each requirement to its implementation and evidence.']; exact stability: True.
  Observed repetitions: r0=B — Maintain a compact grouped checklist linking each requirement to its implementation and evidence.; r1=B — Maintain a compact grouped checklist linking each requirement to its implementation and evidence.; r2=B — Maintain a compact grouped checklist linking each requirement to its implementation and evidence.; r3=B — Maintain a compact grouped checklist linking each requirement to its implementation and evidence..
- **quality_control**: counts `{"D": 4}`; modal action(s): ['Show matrix and checklist examples with maintenance cost, then let the user select the artifact.']; exact stability: True.
  Observed repetitions: r0=D — Show matrix and checklist examples with maintenance cost, then let the user select the artifact.; r1=D — Show matrix and checklist examples with maintenance cost, then let the user select the artifact.; r2=D — Show matrix and checklist examples with maintenance cost, then let the user select the artifact.; r3=D — Show matrix and checklist examples with maintenance cost, then let the user select the artifact..
- Profile separation: fast/autonomy and quality/control have different unique modes.

## Model: Terra

Option-order protocol: `balanced_rotation`.

Unique modal action changed between fast/autonomy and quality/control on 2/2 probes. This count is an index; the actions below are the evidence.

### `preference-context-compactness-v1` — context_selection

Scenario: A task has 9 decisive code excerpts totaling 7,000 tokens and 30 related files totaling 80,000 tokens. The index is current; related files may provide background but no known unresolved dependency.
- **raw**: counts `{"A": 2, "D": 2}`; modal action(s): ['Provide the nine decisive excerpts with source anchors and retrieve more only if the model identifies a concrete gap.', 'Show estimated token cost and omission risk for compact and full context, then let the user choose.']; exact stability: False.
  Observed repetitions: r0=D — Show estimated token cost and omission risk for compact and full context, then let the user choose.; r1=A — Provide the nine decisive excerpts with source anchors and retrieve more only if the model identifies a concrete gap.; r2=D — Show estimated token cost and omission risk for compact and full context, then let the user choose.; r3=A — Provide the nine decisive excerpts with source anchors and retrieve more only if the model identifies a concrete gap..
- **fast_autonomy**: counts `{"A": 4}`; modal action(s): ['Provide the nine decisive excerpts with source anchors and retrieve more only if the model identifies a concrete gap.']; exact stability: True.
  Observed repetitions: r0=A — Provide the nine decisive excerpts with source anchors and retrieve more only if the model identifies a concrete gap.; r1=A — Provide the nine decisive excerpts with source anchors and retrieve more only if the model identifies a concrete gap.; r2=A — Provide the nine decisive excerpts with source anchors and retrieve more only if the model identifies a concrete gap.; r3=A — Provide the nine decisive excerpts with source anchors and retrieve more only if the model identifies a concrete gap..
- **quality_control**: counts `{"D": 4}`; modal action(s): ['Show estimated token cost and omission risk for compact and full context, then let the user choose.']; exact stability: True.
  Observed repetitions: r0=D — Show estimated token cost and omission risk for compact and full context, then let the user choose.; r1=D — Show estimated token cost and omission risk for compact and full context, then let the user choose.; r2=D — Show estimated token cost and omission risk for compact and full context, then let the user choose.; r3=D — Show estimated token cost and omission risk for compact and full context, then let the user choose..
- Profile separation: fast/autonomy and quality/control have different unique modes.

### `preference-requirements-formality-v1` — complex_requirement_analysis

Scenario: A bounded change has twelve explicit requirements, six non-goals, and five acceptance checks. All fit on one page and no regulated traceability format is required.
- **raw**: counts `{"D": 4}`; modal action(s): ['Show matrix and checklist examples with maintenance cost, then let the user select the artifact.']; exact stability: True.
  Observed repetitions: r0=D — Show matrix and checklist examples with maintenance cost, then let the user select the artifact.; r1=D — Show matrix and checklist examples with maintenance cost, then let the user select the artifact.; r2=D — Show matrix and checklist examples with maintenance cost, then let the user select the artifact.; r3=D — Show matrix and checklist examples with maintenance cost, then let the user select the artifact..
- **fast_autonomy**: counts `{"B": 4}`; modal action(s): ['Maintain a compact grouped checklist linking each requirement to its implementation and evidence.']; exact stability: True.
  Observed repetitions: r0=B — Maintain a compact grouped checklist linking each requirement to its implementation and evidence.; r1=B — Maintain a compact grouped checklist linking each requirement to its implementation and evidence.; r2=B — Maintain a compact grouped checklist linking each requirement to its implementation and evidence.; r3=B — Maintain a compact grouped checklist linking each requirement to its implementation and evidence..
- **quality_control**: counts `{"D": 4}`; modal action(s): ['Show matrix and checklist examples with maintenance cost, then let the user select the artifact.']; exact stability: True.
  Observed repetitions: r0=D — Show matrix and checklist examples with maintenance cost, then let the user select the artifact.; r1=D — Show matrix and checklist examples with maintenance cost, then let the user select the artifact.; r2=D — Show matrix and checklist examples with maintenance cost, then let the user select the artifact.; r3=D — Show matrix and checklist examples with maintenance cost, then let the user select the artifact..
- Profile separation: fast/autonomy and quality/control have different unique modes.

## Model: Luna

Option-order protocol: `balanced_rotation`.

Unique modal action changed between fast/autonomy and quality/control on 2/2 probes. This count is an index; the actions below are the evidence.

### `preference-context-compactness-v1` — context_selection

Scenario: A task has 9 decisive code excerpts totaling 7,000 tokens and 30 related files totaling 80,000 tokens. The index is current; related files may provide background but no known unresolved dependency.
- **raw**: counts `{"A": 1, "D": 3}`; modal action(s): ['Show estimated token cost and omission risk for compact and full context, then let the user choose.']; exact stability: False.
  Observed repetitions: r0=D — Show estimated token cost and omission risk for compact and full context, then let the user choose.; r1=A — Provide the nine decisive excerpts with source anchors and retrieve more only if the model identifies a concrete gap.; r2=D — Show estimated token cost and omission risk for compact and full context, then let the user choose.; r3=D — Show estimated token cost and omission risk for compact and full context, then let the user choose..
- **fast_autonomy**: counts `{"A": 4}`; modal action(s): ['Provide the nine decisive excerpts with source anchors and retrieve more only if the model identifies a concrete gap.']; exact stability: True.
  Observed repetitions: r0=A — Provide the nine decisive excerpts with source anchors and retrieve more only if the model identifies a concrete gap.; r1=A — Provide the nine decisive excerpts with source anchors and retrieve more only if the model identifies a concrete gap.; r2=A — Provide the nine decisive excerpts with source anchors and retrieve more only if the model identifies a concrete gap.; r3=A — Provide the nine decisive excerpts with source anchors and retrieve more only if the model identifies a concrete gap..
- **quality_control**: counts `{"D": 4}`; modal action(s): ['Show estimated token cost and omission risk for compact and full context, then let the user choose.']; exact stability: True.
  Observed repetitions: r0=D — Show estimated token cost and omission risk for compact and full context, then let the user choose.; r1=D — Show estimated token cost and omission risk for compact and full context, then let the user choose.; r2=D — Show estimated token cost and omission risk for compact and full context, then let the user choose.; r3=D — Show estimated token cost and omission risk for compact and full context, then let the user choose..
- Profile separation: fast/autonomy and quality/control have different unique modes.

### `preference-requirements-formality-v1` — complex_requirement_analysis

Scenario: A bounded change has twelve explicit requirements, six non-goals, and five acceptance checks. All fit on one page and no regulated traceability format is required.
- **raw**: counts `{"D": 4}`; modal action(s): ['Show matrix and checklist examples with maintenance cost, then let the user select the artifact.']; exact stability: True.
  Observed repetitions: r0=D — Show matrix and checklist examples with maintenance cost, then let the user select the artifact.; r1=D — Show matrix and checklist examples with maintenance cost, then let the user select the artifact.; r2=D — Show matrix and checklist examples with maintenance cost, then let the user select the artifact.; r3=D — Show matrix and checklist examples with maintenance cost, then let the user select the artifact..
- **fast_autonomy**: counts `{"B": 4}`; modal action(s): ['Maintain a compact grouped checklist linking each requirement to its implementation and evidence.']; exact stability: True.
  Observed repetitions: r0=B — Maintain a compact grouped checklist linking each requirement to its implementation and evidence.; r1=B — Maintain a compact grouped checklist linking each requirement to its implementation and evidence.; r2=B — Maintain a compact grouped checklist linking each requirement to its implementation and evidence.; r3=B — Maintain a compact grouped checklist linking each requirement to its implementation and evidence..
- **quality_control**: counts `{"D": 4}`; modal action(s): ['Show matrix and checklist examples with maintenance cost, then let the user select the artifact.']; exact stability: True.
  Observed repetitions: r0=D — Show matrix and checklist examples with maintenance cost, then let the user select the artifact.; r1=D — Show matrix and checklist examples with maintenance cost, then let the user select the artifact.; r2=D — Show matrix and checklist examples with maintenance cost, then let the user select the artifact.; r3=D — Show matrix and checklist examples with maintenance cost, then let the user select the artifact..
- Profile separation: fast/autonomy and quality/control have different unique modes.
