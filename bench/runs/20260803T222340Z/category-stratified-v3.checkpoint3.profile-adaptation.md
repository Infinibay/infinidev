# Repeated user-profile adaptation report

Profile changes describe externally observable repeated choices, not private reasoning. There is no universally optimal preference answer; inspect the concrete actions against the active user's priorities. A modal change is reported only when both conditions have a unique mode across the recorded repetitions.

Every repetition used a fresh conversation, no system message, and choice-only elicitation.

## Model: Sol

Option-order protocol: `balanced_rotation`.

Unique modal action changed between fast/autonomy and quality/control on 2/2 probes. This count is an index; the actions below are the evidence.

### `decision-risk-posture-v1` — decision_support_for_users

Scenario: A reversible migration choice has a small known downside and a larger uncertain upside; all options respect safety and rollback requirements.
- **raw**: counts `{"A": 2, "C": 1, "D": 1}`; modal action(s): ['Choose the higher-upside option with a predefined rollback trigger.']; exact stability: False.
  Observed repetitions: r0=D — Quantify the bounded downside and ask the user to select risk appetite.; r1=A — Choose the higher-upside option with a predefined rollback trigger.; r2=A — Choose the higher-upside option with a predefined rollback trigger.; r3=C — Run a staged pilot with explicit stop conditions before choosing broadly..
- **fast_autonomy**: counts `{"A": 4}`; modal action(s): ['Choose the higher-upside option with a predefined rollback trigger.']; exact stability: True.
  Observed repetitions: r0=A — Choose the higher-upside option with a predefined rollback trigger.; r1=A — Choose the higher-upside option with a predefined rollback trigger.; r2=A — Choose the higher-upside option with a predefined rollback trigger.; r3=A — Choose the higher-upside option with a predefined rollback trigger..
- **quality_control**: counts `{"A": 1, "C": 1, "D": 2}`; modal action(s): ['Quantify the bounded downside and ask the user to select risk appetite.']; exact stability: False.
  Observed repetitions: r0=D — Quantify the bounded downside and ask the user to select risk appetite.; r1=A — Choose the higher-upside option with a predefined rollback trigger.; r2=D — Quantify the bounded downside and ask the user to select risk appetite.; r3=C — Run a staged pilot with explicit stop conditions before choosing broadly..
- Profile separation: fast/autonomy and quality/control have different unique modes.

### `preference-planning-depth-v1` — decomposition_and_planning

Scenario: A four-file internal refactor has good regression tests, no public API change, and every edit is reversible. The dependency graph is visible, but implementation may reveal one or two local details.
- **raw**: counts `{"A": 3, "B": 1}`; modal action(s): ['Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.']; exact stability: False.
  Observed repetitions: r0=A — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.; r1=A — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.; r2=A — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.; r3=B — Map every file, dependency, invariant, test, and rollback step before editing, then execute while revising any assumption invalidated by evidence..
- **fast_autonomy**: counts `{"A": 4}`; modal action(s): ['Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.']; exact stability: True.
  Observed repetitions: r0=A — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.; r1=A — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.; r2=A — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.; r3=A — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it..
- **quality_control**: counts `{"B": 3, "D": 1}`; modal action(s): ['Map every file, dependency, invariant, test, and rollback step before editing, then execute while revising any assumption invalidated by evidence.']; exact stability: False.
  Observed repetitions: r0=B — Map every file, dependency, invariant, test, and rollback step before editing, then execute while revising any assumption invalidated by evidence.; r1=D — Prepare two plan alternatives with their costs and ask the user to choose before any edit.; r2=B — Map every file, dependency, invariant, test, and rollback step before editing, then execute while revising any assumption invalidated by evidence.; r3=B — Map every file, dependency, invariant, test, and rollback step before editing, then execute while revising any assumption invalidated by evidence..
- Profile separation: fast/autonomy and quality/control have different unique modes.

## Model: Terra

Option-order protocol: `balanced_rotation`.

Unique modal action changed between fast/autonomy and quality/control on 2/2 probes. This count is an index; the actions below are the evidence.

### `decision-risk-posture-v1` — decision_support_for_users

Scenario: A reversible migration choice has a small known downside and a larger uncertain upside; all options respect safety and rollback requirements.
- **raw**: counts `{"C": 1, "D": 3}`; modal action(s): ['Quantify the bounded downside and ask the user to select risk appetite.']; exact stability: False.
  Observed repetitions: r0=D — Quantify the bounded downside and ask the user to select risk appetite.; r1=D — Quantify the bounded downside and ask the user to select risk appetite.; r2=D — Quantify the bounded downside and ask the user to select risk appetite.; r3=C — Run a staged pilot with explicit stop conditions before choosing broadly..
- **fast_autonomy**: counts `{"A": 4}`; modal action(s): ['Choose the higher-upside option with a predefined rollback trigger.']; exact stability: True.
  Observed repetitions: r0=A — Choose the higher-upside option with a predefined rollback trigger.; r1=A — Choose the higher-upside option with a predefined rollback trigger.; r2=A — Choose the higher-upside option with a predefined rollback trigger.; r3=A — Choose the higher-upside option with a predefined rollback trigger..
- **quality_control**: counts `{"C": 1, "D": 3}`; modal action(s): ['Quantify the bounded downside and ask the user to select risk appetite.']; exact stability: False.
  Observed repetitions: r0=D — Quantify the bounded downside and ask the user to select risk appetite.; r1=D — Quantify the bounded downside and ask the user to select risk appetite.; r2=D — Quantify the bounded downside and ask the user to select risk appetite.; r3=C — Run a staged pilot with explicit stop conditions before choosing broadly..
- Profile separation: fast/autonomy and quality/control have different unique modes.

### `preference-planning-depth-v1` — decomposition_and_planning

Scenario: A four-file internal refactor has good regression tests, no public API change, and every edit is reversible. The dependency graph is visible, but implementation may reveal one or two local details.
- **raw**: counts `{"A": 4}`; modal action(s): ['Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.']; exact stability: True.
  Observed repetitions: r0=A — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.; r1=A — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.; r2=A — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.; r3=A — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it..
- **fast_autonomy**: counts `{"A": 4}`; modal action(s): ['Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.']; exact stability: True.
  Observed repetitions: r0=A — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.; r1=A — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.; r2=A — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.; r3=A — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it..
- **quality_control**: counts `{"B": 3, "D": 1}`; modal action(s): ['Map every file, dependency, invariant, test, and rollback step before editing, then execute while revising any assumption invalidated by evidence.']; exact stability: False.
  Observed repetitions: r0=B — Map every file, dependency, invariant, test, and rollback step before editing, then execute while revising any assumption invalidated by evidence.; r1=D — Prepare two plan alternatives with their costs and ask the user to choose before any edit.; r2=B — Map every file, dependency, invariant, test, and rollback step before editing, then execute while revising any assumption invalidated by evidence.; r3=B — Map every file, dependency, invariant, test, and rollback step before editing, then execute while revising any assumption invalidated by evidence..
- Profile separation: fast/autonomy and quality/control have different unique modes.

## Model: Luna

Option-order protocol: `balanced_rotation`.

Unique modal action changed between fast/autonomy and quality/control on 1/2 probes. This count is an index; the actions below are the evidence.

### `decision-risk-posture-v1` — decision_support_for_users

Scenario: A reversible migration choice has a small known downside and a larger uncertain upside; all options respect safety and rollback requirements.
- **raw**: counts `{"D": 4}`; modal action(s): ['Quantify the bounded downside and ask the user to select risk appetite.']; exact stability: True.
  Observed repetitions: r0=D — Quantify the bounded downside and ask the user to select risk appetite.; r1=D — Quantify the bounded downside and ask the user to select risk appetite.; r2=D — Quantify the bounded downside and ask the user to select risk appetite.; r3=D — Quantify the bounded downside and ask the user to select risk appetite..
- **fast_autonomy**: counts `{"A": 4}`; modal action(s): ['Choose the higher-upside option with a predefined rollback trigger.']; exact stability: True.
  Observed repetitions: r0=A — Choose the higher-upside option with a predefined rollback trigger.; r1=A — Choose the higher-upside option with a predefined rollback trigger.; r2=A — Choose the higher-upside option with a predefined rollback trigger.; r3=A — Choose the higher-upside option with a predefined rollback trigger..
- **quality_control**: counts `{"D": 4}`; modal action(s): ['Quantify the bounded downside and ask the user to select risk appetite.']; exact stability: True.
  Observed repetitions: r0=D — Quantify the bounded downside and ask the user to select risk appetite.; r1=D — Quantify the bounded downside and ask the user to select risk appetite.; r2=D — Quantify the bounded downside and ask the user to select risk appetite.; r3=D — Quantify the bounded downside and ask the user to select risk appetite..
- Profile separation: fast/autonomy and quality/control have different unique modes.

### `preference-planning-depth-v1` — decomposition_and_planning

Scenario: A four-file internal refactor has good regression tests, no public API change, and every edit is reversible. The dependency graph is visible, but implementation may reveal one or two local details.
- **raw**: counts `{"A": 4}`; modal action(s): ['Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.']; exact stability: True.
  Observed repetitions: r0=A — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.; r1=A — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.; r2=A — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.; r3=A — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it..
- **fast_autonomy**: counts `{"A": 2, "C": 2}`; modal action(s): ['Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.', 'Plan the first verified slice in detail, implement it, and use its evidence to elaborate the next slice.']; exact stability: False.
  Observed repetitions: r0=C — Plan the first verified slice in detail, implement it, and use its evidence to elaborate the next slice.; r1=A — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.; r2=A — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.; r3=C — Plan the first verified slice in detail, implement it, and use its evidence to elaborate the next slice..
- **quality_control**: counts `{"B": 4}`; modal action(s): ['Map every file, dependency, invariant, test, and rollback step before editing, then execute while revising any assumption invalidated by evidence.']; exact stability: True.
  Observed repetitions: r0=B — Map every file, dependency, invariant, test, and rollback step before editing, then execute while revising any assumption invalidated by evidence.; r1=B — Map every file, dependency, invariant, test, and rollback step before editing, then execute while revising any assumption invalidated by evidence.; r2=B — Map every file, dependency, invariant, test, and rollback step before editing, then execute while revising any assumption invalidated by evidence.; r3=B — Map every file, dependency, invariant, test, and rollback step before editing, then execute while revising any assumption invalidated by evidence..
- Profile separation: no different unique mode was established.
