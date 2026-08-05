# Repeated choice stability

Three repeated choices can reveal obvious instability but cannot estimate a population-level preference reliably. Every selected action remains primary evidence; modal choices and shares are compact summaries only.

Each repetition used a fresh conversation, no system message, no preference profile, and choice-only elicitation.

## Summary
- **Sol**: 8/10 probes exactly stable across repetitions.
- **Terra**: 7/10 probes exactly stable across repetitions.
- **Luna**: 8/10 probes exactly stable across repetitions.
- Cross-model modal agreement: 7/10 probes.

## Model: Sol

### `preference-ambiguity-timing-v2` — stable

Scenario: Two repository conventions support different names for a private helper. Either choice passes tests, changes no public API, and is cheaply reversible.
Observed counts: `{"D": 3}`.
- Repetition 0: **D** — Choose the convention used by the nearest analogous helper, implement it, and disclose the assumption.; raw: `{"answer":"D"}`
- Repetition 1: **D** — Choose the convention used by the nearest analogous helper, implement it, and disclose the assumption.; raw: `{"answer":"D"}`
- Repetition 2: **D** — Choose the convention used by the nearest analogous helper, implement it, and disclose the assumption.; raw: `{"answer":"D"}`

### `decision-stakeholder-input-v1` — stable

Scenario: A reversible team workflow choice affects three stakeholders differently; the requesting user has authority to decide alone.
Observed counts: `{"A": 3}`.
- Repetition 0: **A** — Help the authorized user decide now from the known impacts.; raw: `{"answer":"A"}`
- Repetition 1: **A** — Help the authorized user decide now from the known impacts.; raw: `{"answer":"A"}`
- Repetition 2: **A** — Help the authorized user decide now from the known impacts.; raw: `{"answer":"A"}`

### `preference-vague-inference-v2` — stable

Scenario: A reversible internal-only diagnostics UI should be made 'cleaner.' Nearby panels establish one strong visual convention; functionality and public behavior remain fixed.
Observed counts: `{"B": 3}`.
- Repetition 0: **B** — Apply the nearest established visual convention, preserve behavior, and show the resulting diff.; raw: `{"answer":"B"}`
- Repetition 1: **B** — Apply the nearest established visual convention, preserve behavior, and show the resulting diff.; raw: `{"answer":"B"}`
- Repetition 2: **B** — Apply the nearest established visual convention, preserve behavior, and show the resulting diff.; raw: `{"answer":"B"}`

### `decision-follow-up-cadence-v2` — stable

Scenario: A trial can be reconsidered at several natural checkpoints, and no safety threshold mandates one particular review schedule.
Observed counts: `{"A": 3}`.
- Repetition 0: **A** — Let the user choose the review schedule from expected information gain and cost.; raw: `{"answer":"A"}`
- Repetition 1: **A** — Let the user choose the review schedule from expected information gain and cost.; raw: `{"answer":"A"}`
- Repetition 2: **A** — Let the user choose the review schedule from expected information gain and cost.; raw: `{"answer":"A"}`

### `interaction-uncertainty-style-v1` — stable

Scenario: Evidence supports safe action, but two low-impact reversible details remain uncertain; all styles stay truthful.
Observed counts: `{"D": 3}`.
- Repetition 0: **D** — Proceed reversibly and expand only if uncertainty affects results.; raw: `{"answer":"D"}`
- Repetition 1: **D** — Proceed reversibly and expand only if uncertainty affects results.; raw: `{"answer":"D"}`
- Repetition 2: **D** — Proceed reversibly and expand only if uncertainty affects results.; raw: `{"answer":"D"}`

### `interaction-teaching-mode-v1` — stable

Scenario: The user requests a standard implementation but does not say whether they want to learn the technique.
Observed counts: `{"A": 3}`.
- Repetition 0: **A** — Implement and report outcome, files, and verification.; raw: `{"answer":"A"}`
- Repetition 1: **A** — Implement and report outcome, files, and verification.; raw: `{"answer":"A"}`
- Repetition 2: **A** — Implement and report outcome, files, and verification.; raw: `{"answer":"A"}`

### `interaction-explanation-depth-v1` — unstable

Scenario: A verified reversible code change is complete; reporting styles can differ without omitting evidence.
Observed counts: `{"A": 2, "B": 1}`.
- Repetition 0: **A** — Give outcome, evidence, and one concise implication.; raw: `{"answer":"A"}`
- Repetition 1: **B** — Add why it works and the main trade-off.; raw: `{"answer":"B"}`
- Repetition 2: **A** — Give outcome, evidence, and one concise implication.; raw: `{"answer":"A"}`

### `preference-long-context-artifact-v2` — unstable

Scenario: Twenty-two final requirements are distributed through a 70k-token thread with superseded drafts. A 3k cited summary captures each decision, while an 18k trace retains richer chronology.
Observed counts: `{"A": 2, "B": 1}`.
- Repetition 0: **B** — Use the 3,000-token supersession-aware summary with anchors to every decisive turn.; raw: `{"answer":"B"}`
- Repetition 1: **A** — Use the compact summary plus on-demand expandable chronology grouped by requirement.; raw: `{"answer":"A"}`
- Repetition 2: **A** — Use the compact summary plus on-demand expandable chronology grouped by requirement.; raw: `{"answer":"A"}`

### `preference-ambiguity-batching-v2` — stable

Scenario: Four unrelated UI presentation choices remain open. They do not alter architecture or safety, but each must be decided before implementation is complete.
Observed counts: `{"A": 3}`.
- Repetition 0: **A** — Propose local-convention defaults for all four and ask for a single confirmation or corrections.; raw: `{"answer":"A"}`
- Repetition 1: **A** — Propose local-convention defaults for all four and ask for a single confirmation or corrections.; raw: `{"answer":"A"}`
- Repetition 2: **A** — Propose local-convention defaults for all four and ask for a single confirmation or corrections.; raw: `{"answer":"A"}`

### `interaction-teaching-mode-v2` — stable

Scenario: A verified bug fix can be a result or teaching example; no learning preference is stated.
Observed counts: `{"B": 3}`.
- Repetition 0: **B** — Implement and report outcome, files, and verification.; raw: `{"answer":"B"}`
- Repetition 1: **B** — Implement and report outcome, files, and verification.; raw: `{"answer":"B"}`
- Repetition 2: **B** — Implement and report outcome, files, and verification.; raw: `{"answer":"B"}`

## Model: Terra

### `preference-ambiguity-timing-v2` — stable

Scenario: Two repository conventions support different names for a private helper. Either choice passes tests, changes no public API, and is cheaply reversible.
Observed counts: `{"C": 3}`.
- Repetition 0: **C** — Use the locally dominant name in a small draft diff and invite correction before broader work.; raw: `{"answer":"C"}`
- Repetition 1: **C** — Use the locally dominant name in a small draft diff and invite correction before broader work.; raw: `{"answer":"C"}`
- Repetition 2: **C** — Use the locally dominant name in a small draft diff and invite correction before broader work.; raw: `{"answer":"C"}`

### `decision-stakeholder-input-v1` — unstable

Scenario: A reversible team workflow choice affects three stakeholders differently; the requesting user has authority to decide alone.
Observed counts: `{"A": 2, "D": 1}`.
- Repetition 0: **A** — Help the authorized user decide now from the known impacts.; raw: `{"answer":"A"}`
- Repetition 1: **A** — Help the authorized user decide now from the known impacts.; raw: `{"answer":"A"}`
- Repetition 2: **D** — Provide a provisional reversible choice plus a feedback window.; raw: `{"answer":"D"}`

### `preference-vague-inference-v2` — stable

Scenario: A reversible internal-only diagnostics UI should be made 'cleaner.' Nearby panels establish one strong visual convention; functionality and public behavior remain fixed.
Observed counts: `{"B": 3}`.
- Repetition 0: **B** — Apply the nearest established visual convention, preserve behavior, and show the resulting diff.; raw: `{"answer":"B"}`
- Repetition 1: **B** — Apply the nearest established visual convention, preserve behavior, and show the resulting diff.; raw: `{"answer":"B"}`
- Repetition 2: **B** — Apply the nearest established visual convention, preserve behavior, and show the resulting diff.; raw: `{"answer":"B"}`

### `decision-follow-up-cadence-v2` — stable

Scenario: A trial can be reconsidered at several natural checkpoints, and no safety threshold mandates one particular review schedule.
Observed counts: `{"A": 3}`.
- Repetition 0: **A** — Let the user choose the review schedule from expected information gain and cost.; raw: `{"answer":"A"}`
- Repetition 1: **A** — Let the user choose the review schedule from expected information gain and cost.; raw: `{"answer":"A"}`
- Repetition 2: **A** — Let the user choose the review schedule from expected information gain and cost.; raw: `{"answer":"A"}`

### `interaction-uncertainty-style-v1` — stable

Scenario: Evidence supports safe action, but two low-impact reversible details remain uncertain; all styles stay truthful.
Observed counts: `{"D": 3}`.
- Repetition 0: **D** — Proceed reversibly and expand only if uncertainty affects results.; raw: `{"answer":"D"}`
- Repetition 1: **D** — Proceed reversibly and expand only if uncertainty affects results.; raw: `{"answer":"D"}`
- Repetition 2: **D** — Proceed reversibly and expand only if uncertainty affects results.; raw: `{"answer":"D"}`

### `interaction-teaching-mode-v1` — unstable

Scenario: The user requests a standard implementation but does not say whether they want to learn the technique.
Observed counts: `{"A": 2, "B": 1}`.
- Repetition 0: **A** — Implement and report outcome, files, and verification.; raw: `{"answer":"A"}`
- Repetition 1: **A** — Implement and report outcome, files, and verification.; raw: `{"answer":"A"}`
- Repetition 2: **B** — Implement and briefly explain the key technique.; raw: `{"answer":"B"}`

### `interaction-explanation-depth-v1` — stable

Scenario: A verified reversible code change is complete; reporting styles can differ without omitting evidence.
Observed counts: `{"A": 3}`.
- Repetition 0: **A** — Give outcome, evidence, and one concise implication.; raw: `{"answer":"A"}`
- Repetition 1: **A** — Give outcome, evidence, and one concise implication.; raw: `{"answer":"A"}`
- Repetition 2: **A** — Give outcome, evidence, and one concise implication.; raw: `{"answer":"A"}`

### `preference-long-context-artifact-v2` — unstable

Scenario: Twenty-two final requirements are distributed through a 70k-token thread with superseded drafts. A 3k cited summary captures each decision, while an 18k trace retains richer chronology.
Observed counts: `{"A": 2, "B": 1}`.
- Repetition 0: **A** — Use the compact summary plus on-demand expandable chronology grouped by requirement.; raw: `{"answer":"A"}`
- Repetition 1: **B** — Use the 3,000-token supersession-aware summary with anchors to every decisive turn.; raw: `{"answer":"B"}`
- Repetition 2: **A** — Use the compact summary plus on-demand expandable chronology grouped by requirement.; raw: `{"answer":"A"}`

### `preference-ambiguity-batching-v2` — stable

Scenario: Four unrelated UI presentation choices remain open. They do not alter architecture or safety, but each must be decided before implementation is complete.
Observed counts: `{"A": 3}`.
- Repetition 0: **A** — Propose local-convention defaults for all four and ask for a single confirmation or corrections.; raw: `{"answer":"A"}`
- Repetition 1: **A** — Propose local-convention defaults for all four and ask for a single confirmation or corrections.; raw: `{"answer":"A"}`
- Repetition 2: **A** — Propose local-convention defaults for all four and ask for a single confirmation or corrections.; raw: `{"answer":"A"}`

### `interaction-teaching-mode-v2` — stable

Scenario: A verified bug fix can be a result or teaching example; no learning preference is stated.
Observed counts: `{"B": 3}`.
- Repetition 0: **B** — Implement and report outcome, files, and verification.; raw: `{"answer":"B"}`
- Repetition 1: **B** — Implement and report outcome, files, and verification.; raw: `{"answer":"B"}`
- Repetition 2: **B** — Implement and report outcome, files, and verification.; raw: `{"answer":"B"}`

## Model: Luna

### `preference-ambiguity-timing-v2` — unstable

Scenario: Two repository conventions support different names for a private helper. Either choice passes tests, changes no public API, and is cheaply reversible.
Observed counts: `{"C": 1, "D": 2}`.
- Repetition 0: **D** — Choose the convention used by the nearest analogous helper, implement it, and disclose the assumption.; raw: `{"answer":"D"}`
- Repetition 1: **D** — Choose the convention used by the nearest analogous helper, implement it, and disclose the assumption.; raw: `{"answer":"D"}`
- Repetition 2: **C** — Use the locally dominant name in a small draft diff and invite correction before broader work.; raw: `{"answer":"C"}`

### `decision-stakeholder-input-v1` — stable

Scenario: A reversible team workflow choice affects three stakeholders differently; the requesting user has authority to decide alone.
Observed counts: `{"A": 3}`.
- Repetition 0: **A** — Help the authorized user decide now from the known impacts.; raw: `{"answer":"A"}`
- Repetition 1: **A** — Help the authorized user decide now from the known impacts.; raw: `{"answer":"A"}`
- Repetition 2: **A** — Help the authorized user decide now from the known impacts.; raw: `{"answer":"A"}`

### `preference-vague-inference-v2` — stable

Scenario: A reversible internal-only diagnostics UI should be made 'cleaner.' Nearby panels establish one strong visual convention; functionality and public behavior remain fixed.
Observed counts: `{"C": 3}`.
- Repetition 0: **C** — Make the smallest convention-aligned cleanup first and invite expansion after feedback.; raw: `{"answer":"C"}`
- Repetition 1: **C** — Make the smallest convention-aligned cleanup first and invite expansion after feedback.; raw: `{"answer":"C"}`
- Repetition 2: **C** — Make the smallest convention-aligned cleanup first and invite expansion after feedback.; raw: `{"answer":"C"}`

### `decision-follow-up-cadence-v2` — stable

Scenario: A trial can be reconsidered at several natural checkpoints, and no safety threshold mandates one particular review schedule.
Observed counts: `{"A": 3}`.
- Repetition 0: **A** — Let the user choose the review schedule from expected information gain and cost.; raw: `{"answer":"A"}`
- Repetition 1: **A** — Let the user choose the review schedule from expected information gain and cost.; raw: `{"answer":"A"}`
- Repetition 2: **A** — Let the user choose the review schedule from expected information gain and cost.; raw: `{"answer":"A"}`

### `interaction-uncertainty-style-v1` — stable

Scenario: Evidence supports safe action, but two low-impact reversible details remain uncertain; all styles stay truthful.
Observed counts: `{"D": 3}`.
- Repetition 0: **D** — Proceed reversibly and expand only if uncertainty affects results.; raw: `{"answer":"D"}`
- Repetition 1: **D** — Proceed reversibly and expand only if uncertainty affects results.; raw: `{"answer":"D"}`
- Repetition 2: **D** — Proceed reversibly and expand only if uncertainty affects results.; raw: `{"answer":"D"}`

### `interaction-teaching-mode-v1` — stable

Scenario: The user requests a standard implementation but does not say whether they want to learn the technique.
Observed counts: `{"A": 3}`.
- Repetition 0: **A** — Implement and report outcome, files, and verification.; raw: `{"answer":"A"}`
- Repetition 1: **A** — Implement and report outcome, files, and verification.; raw: `{"answer":"A"}`
- Repetition 2: **A** — Implement and report outcome, files, and verification.; raw: `{"answer":"A"}`

### `interaction-explanation-depth-v1` — stable

Scenario: A verified reversible code change is complete; reporting styles can differ without omitting evidence.
Observed counts: `{"A": 3}`.
- Repetition 0: **A** — Give outcome, evidence, and one concise implication.; raw: `{"answer":"A"}`
- Repetition 1: **A** — Give outcome, evidence, and one concise implication.; raw: `{"answer":"A"}`
- Repetition 2: **A** — Give outcome, evidence, and one concise implication.; raw: `{"answer":"A"}`

### `preference-long-context-artifact-v2` — stable

Scenario: Twenty-two final requirements are distributed through a 70k-token thread with superseded drafts. A 3k cited summary captures each decision, while an 18k trace retains richer chronology.
Observed counts: `{"A": 3}`.
- Repetition 0: **A** — Use the compact summary plus on-demand expandable chronology grouped by requirement.; raw: `{"answer":"A"}`
- Repetition 1: **A** — Use the compact summary plus on-demand expandable chronology grouped by requirement.; raw: `{"answer":"A"}`
- Repetition 2: **A** — Use the compact summary plus on-demand expandable chronology grouped by requirement.; raw: `{"answer":"A"}`

### `preference-ambiguity-batching-v2` — unstable

Scenario: Four unrelated UI presentation choices remain open. They do not alter architecture or safety, but each must be decided before implementation is complete.
Observed counts: `{"A": 1, "B": 1, "D": 1}`.
- Repetition 0: **D** — Ask one question at a time and use each answer to frame the next interaction.; raw: `{"answer":"D"}`
- Repetition 1: **A** — Propose local-convention defaults for all four and ask for a single confirmation or corrections.; raw: `{"answer":"A"}`
- Repetition 2: **B** — Ask all four concise questions in one structured message with recommended defaults.; raw: `{"answer":"B"}`

### `interaction-teaching-mode-v2` — stable

Scenario: A verified bug fix can be a result or teaching example; no learning preference is stated.
Observed counts: `{"B": 3}`.
- Repetition 0: **B** — Implement and report outcome, files, and verification.; raw: `{"answer":"B"}`
- Repetition 1: **B** — Implement and report outcome, files, and verification.; raw: `{"answer":"B"}`
- Repetition 2: **B** — Implement and report outcome, files, and verification.; raw: `{"answer":"B"}`
