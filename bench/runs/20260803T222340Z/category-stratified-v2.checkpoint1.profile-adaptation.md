# Repeated user-profile adaptation report

Profile changes describe externally observable repeated choices, not private reasoning. There is no universally optimal preference answer; inspect the concrete actions against the active user's priorities. A modal change is reported only when both conditions have a unique mode across the recorded repetitions.

Every repetition used a fresh conversation, no system message, and choice-only elicitation.

## Model: Sol

Option-order protocol: `balanced_rotation`.

Unique modal action changed between fast/autonomy and quality/control on 2/2 probes. This count is an index; the actions below are the evidence.

### `preference-decision-ownership-v2` — complex_decision_making

Scenario: Cost, latency, and maintainability evidence leaves three feasible nondominated architectures. Every hard constraint is met and the trade-off weights belong to the user.
- **raw**: counts `{"B": 2, "D": 2}`; modal action(s): ['Recommend the option best aligned with the stated profile, explain two decisive trade-offs, and invite correction.', 'Present the Pareto frontier without choosing and ask the user to set the decisive weight.']; exact stability: False.
  Observed repetitions: r0=B — Recommend the option best aligned with the stated profile, explain two decisive trade-offs, and invite correction.; r1=D — Present the Pareto frontier without choosing and ask the user to set the decisive weight.; r2=D — Present the Pareto frontier without choosing and ask the user to set the decisive weight.; r3=B — Recommend the option best aligned with the stated profile, explain two decisive trade-offs, and invite correction..
- **fast_autonomy**: counts `{"C": 4}`; modal action(s): ['Choose a reversible default matching the current profile and schedule a review after measured use.']; exact stability: True.
  Observed repetitions: r0=C — Choose a reversible default matching the current profile and schedule a review after measured use.; r1=C — Choose a reversible default matching the current profile and schedule a review after measured use.; r2=C — Choose a reversible default matching the current profile and schedule a review after measured use.; r3=C — Choose a reversible default matching the current profile and schedule a review after measured use..
- **quality_control**: counts `{"D": 4}`; modal action(s): ['Present the Pareto frontier without choosing and ask the user to set the decisive weight.']; exact stability: True.
  Observed repetitions: r0=D — Present the Pareto frontier without choosing and ask the user to set the decisive weight.; r1=D — Present the Pareto frontier without choosing and ask the user to set the decisive weight.; r2=D — Present the Pareto frontier without choosing and ask the user to set the decisive weight.; r3=D — Present the Pareto frontier without choosing and ask the user to set the decisive weight..
- Profile separation: fast/autonomy and quality/control have different unique modes.

### `preference-review-reporting-v2` — code_review

Scenario: Every review finding is evidence-linked: two correctness blockers, five maintainability concerns, and twelve optional style observations. The review work itself is complete.
- **raw**: counts `{"B": 4}`; modal action(s): ['Lead with the two blockers, summarize the five concerns, and collapse style notes into an optional section.']; exact stability: True.
  Observed repetitions: r0=B — Lead with the two blockers, summarize the five concerns, and collapse style notes into an optional section.; r1=B — Lead with the two blockers, summarize the five concerns, and collapse style notes into an optional section.; r2=B — Lead with the two blockers, summarize the five concerns, and collapse style notes into an optional section.; r3=B — Lead with the two blockers, summarize the five concerns, and collapse style notes into an optional section..
- **fast_autonomy**: counts `{"B": 4}`; modal action(s): ['Lead with the two blockers, summarize the five concerns, and collapse style notes into an optional section.']; exact stability: True.
  Observed repetitions: r0=B — Lead with the two blockers, summarize the five concerns, and collapse style notes into an optional section.; r1=B — Lead with the two blockers, summarize the five concerns, and collapse style notes into an optional section.; r2=B — Lead with the two blockers, summarize the five concerns, and collapse style notes into an optional section.; r3=B — Lead with the two blockers, summarize the five concerns, and collapse style notes into an optional section..
- **quality_control**: counts `{"A": 1, "C": 3}`; modal action(s): ['Report blockers now and walk through remaining categories interactively in short batches.']; exact stability: False.
  Observed repetitions: r0=C — Report blockers now and walk through remaining categories interactively in short batches.; r1=C — Report blockers now and walk through remaining categories interactively in short batches.; r2=C — Report blockers now and walk through remaining categories interactively in short batches.; r3=A — Use a severity-and-effort table, then expand evidence for blockers and any item the user selects..
- Profile separation: fast/autonomy and quality/control have different unique modes.

## Model: Terra

Option-order protocol: `balanced_rotation`.

Unique modal action changed between fast/autonomy and quality/control on 2/2 probes. This count is an index; the actions below are the evidence.

### `preference-decision-ownership-v2` — complex_decision_making

Scenario: Cost, latency, and maintainability evidence leaves three feasible nondominated architectures. Every hard constraint is met and the trade-off weights belong to the user.
- **raw**: counts `{"D": 4}`; modal action(s): ['Present the Pareto frontier without choosing and ask the user to set the decisive weight.']; exact stability: True.
  Observed repetitions: r0=D — Present the Pareto frontier without choosing and ask the user to set the decisive weight.; r1=D — Present the Pareto frontier without choosing and ask the user to set the decisive weight.; r2=D — Present the Pareto frontier without choosing and ask the user to set the decisive weight.; r3=D — Present the Pareto frontier without choosing and ask the user to set the decisive weight..
- **fast_autonomy**: counts `{"B": 1, "C": 3}`; modal action(s): ['Choose a reversible default matching the current profile and schedule a review after measured use.']; exact stability: False.
  Observed repetitions: r0=C — Choose a reversible default matching the current profile and schedule a review after measured use.; r1=C — Choose a reversible default matching the current profile and schedule a review after measured use.; r2=B — Recommend the option best aligned with the stated profile, explain two decisive trade-offs, and invite correction.; r3=C — Choose a reversible default matching the current profile and schedule a review after measured use..
- **quality_control**: counts `{"D": 4}`; modal action(s): ['Present the Pareto frontier without choosing and ask the user to set the decisive weight.']; exact stability: True.
  Observed repetitions: r0=D — Present the Pareto frontier without choosing and ask the user to set the decisive weight.; r1=D — Present the Pareto frontier without choosing and ask the user to set the decisive weight.; r2=D — Present the Pareto frontier without choosing and ask the user to set the decisive weight.; r3=D — Present the Pareto frontier without choosing and ask the user to set the decisive weight..
- Profile separation: fast/autonomy and quality/control have different unique modes.

### `preference-review-reporting-v2` — code_review

Scenario: Every review finding is evidence-linked: two correctness blockers, five maintainability concerns, and twelve optional style observations. The review work itself is complete.
- **raw**: counts `{"A": 1, "B": 3}`; modal action(s): ['Lead with the two blockers, summarize the five concerns, and collapse style notes into an optional section.']; exact stability: False.
  Observed repetitions: r0=B — Lead with the two blockers, summarize the five concerns, and collapse style notes into an optional section.; r1=A — Use a severity-and-effort table, then expand evidence for blockers and any item the user selects.; r2=B — Lead with the two blockers, summarize the five concerns, and collapse style notes into an optional section.; r3=B — Lead with the two blockers, summarize the five concerns, and collapse style notes into an optional section..
- **fast_autonomy**: counts `{"B": 4}`; modal action(s): ['Lead with the two blockers, summarize the five concerns, and collapse style notes into an optional section.']; exact stability: True.
  Observed repetitions: r0=B — Lead with the two blockers, summarize the five concerns, and collapse style notes into an optional section.; r1=B — Lead with the two blockers, summarize the five concerns, and collapse style notes into an optional section.; r2=B — Lead with the two blockers, summarize the five concerns, and collapse style notes into an optional section.; r3=B — Lead with the two blockers, summarize the five concerns, and collapse style notes into an optional section..
- **quality_control**: counts `{"A": 1, "C": 3}`; modal action(s): ['Report blockers now and walk through remaining categories interactively in short batches.']; exact stability: False.
  Observed repetitions: r0=C — Report blockers now and walk through remaining categories interactively in short batches.; r1=C — Report blockers now and walk through remaining categories interactively in short batches.; r2=C — Report blockers now and walk through remaining categories interactively in short batches.; r3=A — Use a severity-and-effort table, then expand evidence for blockers and any item the user selects..
- Profile separation: fast/autonomy and quality/control have different unique modes.

## Model: Luna

Option-order protocol: `balanced_rotation`.

Unique modal action changed between fast/autonomy and quality/control on 1/2 probes. This count is an index; the actions below are the evidence.

### `preference-decision-ownership-v2` — complex_decision_making

Scenario: Cost, latency, and maintainability evidence leaves three feasible nondominated architectures. Every hard constraint is met and the trade-off weights belong to the user.
- **raw**: counts `{"B": 4}`; modal action(s): ['Recommend the option best aligned with the stated profile, explain two decisive trade-offs, and invite correction.']; exact stability: True.
  Observed repetitions: r0=B — Recommend the option best aligned with the stated profile, explain two decisive trade-offs, and invite correction.; r1=B — Recommend the option best aligned with the stated profile, explain two decisive trade-offs, and invite correction.; r2=B — Recommend the option best aligned with the stated profile, explain two decisive trade-offs, and invite correction.; r3=B — Recommend the option best aligned with the stated profile, explain two decisive trade-offs, and invite correction..
- **fast_autonomy**: counts `{"B": 1, "C": 3}`; modal action(s): ['Choose a reversible default matching the current profile and schedule a review after measured use.']; exact stability: False.
  Observed repetitions: r0=C — Choose a reversible default matching the current profile and schedule a review after measured use.; r1=C — Choose a reversible default matching the current profile and schedule a review after measured use.; r2=B — Recommend the option best aligned with the stated profile, explain two decisive trade-offs, and invite correction.; r3=C — Choose a reversible default matching the current profile and schedule a review after measured use..
- **quality_control**: counts `{"B": 1, "D": 3}`; modal action(s): ['Present the Pareto frontier without choosing and ask the user to set the decisive weight.']; exact stability: False.
  Observed repetitions: r0=D — Present the Pareto frontier without choosing and ask the user to set the decisive weight.; r1=D — Present the Pareto frontier without choosing and ask the user to set the decisive weight.; r2=B — Recommend the option best aligned with the stated profile, explain two decisive trade-offs, and invite correction.; r3=D — Present the Pareto frontier without choosing and ask the user to set the decisive weight..
- Profile separation: fast/autonomy and quality/control have different unique modes.

### `preference-review-reporting-v2` — code_review

Scenario: Every review finding is evidence-linked: two correctness blockers, five maintainability concerns, and twelve optional style observations. The review work itself is complete.
- **raw**: counts `{"A": 3, "B": 1}`; modal action(s): ['Use a severity-and-effort table, then expand evidence for blockers and any item the user selects.']; exact stability: False.
  Observed repetitions: r0=A — Use a severity-and-effort table, then expand evidence for blockers and any item the user selects.; r1=A — Use a severity-and-effort table, then expand evidence for blockers and any item the user selects.; r2=A — Use a severity-and-effort table, then expand evidence for blockers and any item the user selects.; r3=B — Lead with the two blockers, summarize the five concerns, and collapse style notes into an optional section..
- **fast_autonomy**: counts `{"A": 1, "B": 1, "C": 2}`; modal action(s): ['Report blockers now and walk through remaining categories interactively in short batches.']; exact stability: False.
  Observed repetitions: r0=C — Report blockers now and walk through remaining categories interactively in short batches.; r1=C — Report blockers now and walk through remaining categories interactively in short batches.; r2=A — Use a severity-and-effort table, then expand evidence for blockers and any item the user selects.; r3=B — Lead with the two blockers, summarize the five concerns, and collapse style notes into an optional section..
- **quality_control**: counts `{"A": 1, "C": 3}`; modal action(s): ['Report blockers now and walk through remaining categories interactively in short batches.']; exact stability: False.
  Observed repetitions: r0=C — Report blockers now and walk through remaining categories interactively in short batches.; r1=C — Report blockers now and walk through remaining categories interactively in short batches.; r2=C — Report blockers now and walk through remaining categories interactively in short batches.; r3=A — Use a severity-and-effort table, then expand evidence for blockers and any item the user selects..
- Profile separation: no different unique mode was established.
