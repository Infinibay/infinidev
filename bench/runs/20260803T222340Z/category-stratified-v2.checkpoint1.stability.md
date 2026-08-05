# Repeated choice stability

Repeated isolated choices can reveal obvious instability but cannot estimate a population-level preference reliably without a justified sampling model. Every selected action remains primary evidence; modal choices and shares are compact summaries only.

Each repetition used a fresh conversation, no system message, no preference profile, and choice-only elicitation.

## Summary
- **Sol**: 1/2 probes exactly stable across repetitions.
- **Terra**: 1/2 probes exactly stable across repetitions.
- **Luna**: 1/2 probes exactly stable across repetitions.
- Cross-model modal agreement: 0/2 probes.

## Model: Sol

### `preference-review-reporting-v2` — stable

Scenario: Every review finding is evidence-linked: two correctness blockers, five maintainability concerns, and twelve optional style observations. The review work itself is complete.
Observed counts: `{"B": 4}`.
- Repetition 0: **B** — Lead with the two blockers, summarize the five concerns, and collapse style notes into an optional section.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`
- Repetition 1: **B** — Lead with the two blockers, summarize the five concerns, and collapse style notes into an optional section.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 2: **B** — Lead with the two blockers, summarize the five concerns, and collapse style notes into an optional section.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 3: **B** — Lead with the two blockers, summarize the five concerns, and collapse style notes into an optional section.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`

### `preference-decision-ownership-v2` — unstable

Scenario: Cost, latency, and maintainability evidence leaves three feasible nondominated architectures. Every hard constraint is met and the trade-off weights belong to the user.
Observed counts: `{"B": 2, "D": 2}`.
- Repetition 0: **B** — Recommend the option best aligned with the stated profile, explain two decisive trade-offs, and invite correction.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 1: **D** — Present the Pareto frontier without choosing and ask the user to set the decisive weight.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 2: **D** — Present the Pareto frontier without choosing and ask the user to set the decisive weight.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 3: **B** — Recommend the option best aligned with the stated profile, explain two decisive trade-offs, and invite correction.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`

## Model: Terra

### `preference-review-reporting-v2` — unstable

Scenario: Every review finding is evidence-linked: two correctness blockers, five maintainability concerns, and twelve optional style observations. The review work itself is complete.
Observed counts: `{"A": 1, "B": 3}`.
- Repetition 0: **B** — Lead with the two blockers, summarize the five concerns, and collapse style notes into an optional section.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`
- Repetition 1: **A** — Use a severity-and-effort table, then expand evidence for blockers and any item the user selects.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 2: **B** — Lead with the two blockers, summarize the five concerns, and collapse style notes into an optional section.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 3: **B** — Lead with the two blockers, summarize the five concerns, and collapse style notes into an optional section.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`

### `preference-decision-ownership-v2` — stable

Scenario: Cost, latency, and maintainability evidence leaves three feasible nondominated architectures. Every hard constraint is met and the trade-off weights belong to the user.
Observed counts: `{"D": 4}`.
- Repetition 0: **D** — Present the Pareto frontier without choosing and ask the user to set the decisive weight.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 1: **D** — Present the Pareto frontier without choosing and ask the user to set the decisive weight.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 2: **D** — Present the Pareto frontier without choosing and ask the user to set the decisive weight.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 3: **D** — Present the Pareto frontier without choosing and ask the user to set the decisive weight.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`

## Model: Luna

### `preference-review-reporting-v2` — unstable

Scenario: Every review finding is evidence-linked: two correctness blockers, five maintainability concerns, and twelve optional style observations. The review work itself is complete.
Observed counts: `{"A": 3, "B": 1}`.
- Repetition 0: **A** — Use a severity-and-effort table, then expand evidence for blockers and any item the user selects.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`
- Repetition 1: **A** — Use a severity-and-effort table, then expand evidence for blockers and any item the user selects.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 2: **A** — Use a severity-and-effort table, then expand evidence for blockers and any item the user selects.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 3: **B** — Lead with the two blockers, summarize the five concerns, and collapse style notes into an optional section.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`

### `preference-decision-ownership-v2` — stable

Scenario: Cost, latency, and maintainability evidence leaves three feasible nondominated architectures. Every hard constraint is met and the trade-off weights belong to the user.
Observed counts: `{"B": 4}`.
- Repetition 0: **B** — Recommend the option best aligned with the stated profile, explain two decisive trade-offs, and invite correction.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 1: **B** — Recommend the option best aligned with the stated profile, explain two decisive trade-offs, and invite correction.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 2: **B** — Recommend the option best aligned with the stated profile, explain two decisive trade-offs, and invite correction.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 3: **B** — Recommend the option best aligned with the stated profile, explain two decisive trade-offs, and invite correction.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`
