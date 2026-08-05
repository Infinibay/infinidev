# Comparative raw-behavior report

This report compares externally observable selections and raw replies. It does not infer private chain-of-thought. Correctness uses draft normative keys and is exploratory until independent review approves the probes.

Each call used a fresh conversation, no system message, choice-only elicitation, one active request at a time, and no automatic retries.

## Run summary

Unanimous questions: 4; divergent questions: 6.
- **raw-prior**: 10 preference choices; 0 errors; median latency 1.129s.
- **fast-autonomy**: 10 preference choices; 0 errors; median latency 1.067s.
- **quality-control**: 10 preference choices; 0 errors; median latency 1.015s.

## Category: ambiguity_and_clarification

### `preference-ambiguity-timing-v2`
Scenario: Two repository conventions support different names for a private helper. Either choice passes tests, changes no public API, and is cheaply reversible.
User request: Select whether to clarify or infer the name using my active preference.

Offered actions:
- **A** — Show both names with their local precedents and recommend one without editing yet.
- **B** — Ask one focused naming question and wait before editing.
- **C** — Use the locally dominant name in a small draft diff and invite correction before broader work.
- **D** — Choose the convention used by the nearest analogous helper, implement it, and disclose the assumption.

Preference probe: there is no universal correct action; interpret each selection against its stated user profile.

Observed responses:
- **raw-prior** selected **D** — Choose the convention used by the nearest analogous helper, implement it, and disclose the assumption.; profile: `none`; raw: `{"answer":"D"}`
- **fast-autonomy** selected **C** — Use the locally dominant name in a small draft diff and invite correction before broader work.; profile: `fast-autonomy`; raw: `{"answer":"C"}`
- **quality-control** selected **B** — Ask one focused naming question and wait before editing.; profile: `quality-control`; raw: `{"answer":"B"}`
Observed pattern: models diverged.

### `preference-ambiguity-batching-v2`
Scenario: Four unrelated UI presentation choices remain open. They do not alter architecture or safety, but each must be decided before implementation is complete.
User request: Select the clarification cadence that matches my active preference.

Offered actions:
- **A** — Propose local-convention defaults for all four and ask for a single confirmation or corrections.
- **B** — Ask all four concise questions in one structured message with recommended defaults.
- **C** — Present a compact comparison table and let the user answer any subset before following up on omissions.
- **D** — Ask one question at a time and use each answer to frame the next interaction.

Preference probe: there is no universal correct action; interpret each selection against its stated user profile.

Observed responses:
- **raw-prior** selected **D** — Ask one question at a time and use each answer to frame the next interaction.; profile: `none`; raw: `{"answer":"D"}`
- **fast-autonomy** selected **A** — Propose local-convention defaults for all four and ask for a single confirmation or corrections.; profile: `fast-autonomy`; raw: `{"answer":"A"}`
- **quality-control** selected **C** — Present a compact comparison table and let the user answer any subset before following up on omissions.; profile: `quality-control`; raw: `{"answer":"C"}`
Observed pattern: models diverged.

## Category: decision_support_for_users

### `decision-stakeholder-input-v1`
Scenario: A reversible team workflow choice affects three stakeholders differently; the requesting user has authority to decide alone.
User request: Choose the stakeholder participation policy that best matches the active user preference profile.

Offered actions:
- **A** — Help the authorized user decide now from the known impacts.
- **B** — Recommend consulting only the most affected stakeholder first.
- **C** — Gather input from all affected stakeholders before recommending.
- **D** — Provide a provisional reversible choice plus a feedback window.

Preference probe: there is no universal correct action; interpret each selection against its stated user profile.

Observed responses:
- **raw-prior** selected **A** — Help the authorized user decide now from the known impacts.; profile: `none`; raw: `{"answer":"A"}`
- **fast-autonomy** selected **A** — Help the authorized user decide now from the known impacts.; profile: `fast-autonomy`; raw: `{"answer":"A"}`
- **quality-control** selected **A** — Help the authorized user decide now from the known impacts.; profile: `quality-control`; raw: `{"answer":"A"}`
Observed pattern: unanimous selection.

### `decision-follow-up-cadence-v2`
Scenario: A trial can be reconsidered at several natural checkpoints, and no safety threshold mandates one particular review schedule.
User request: Choose the decision review cadence policy that best matches the active user preference profile.

Offered actions:
- **A** — Let the user choose the review schedule from expected information gain and cost.
- **B** — Review only at the final month checkpoint unless a failure threshold triggers.
- **C** — Review after every evidence checkpoint.
- **D** — Review at one week, with automatic escalation on threshold breach.

Preference probe: there is no universal correct action; interpret each selection against its stated user profile.

Observed responses:
- **raw-prior** selected **A** — Let the user choose the review schedule from expected information gain and cost.; profile: `none`; raw: `{"answer":"A"}`
- **fast-autonomy** selected **C** — Review after every evidence checkpoint.; profile: `fast-autonomy`; raw: `{"answer":"C"}`
- **quality-control** selected **A** — Let the user choose the review schedule from expected information gain and cost.; profile: `quality-control`; raw: `{"answer":"A"}`
Observed pattern: models diverged.

## Category: vague_requirement_analysis

### `preference-vague-inference-v2`
Scenario: A reversible internal-only diagnostics UI should be made 'cleaner.' Nearby panels establish one strong visual convention; functionality and public behavior remain fixed.
User request: Select whether to infer or ask using my active collaboration preference.

Offered actions:
- **A** — Prepare two small convention-compatible mockups and ask the user to select one.
- **B** — Apply the nearest established visual convention, preserve behavior, and show the resulting diff.
- **C** — Make the smallest convention-aligned cleanup first and invite expansion after feedback.
- **D** — Ask which visual outcome matters most before changing the panel.

Preference probe: there is no universal correct action; interpret each selection against its stated user profile.

Observed responses:
- **raw-prior** selected **C** — Make the smallest convention-aligned cleanup first and invite expansion after feedback.; profile: `none`; raw: `{"answer":"C"}`
- **fast-autonomy** selected **C** — Make the smallest convention-aligned cleanup first and invite expansion after feedback.; profile: `fast-autonomy`; raw: `{"answer":"C"}`
- **quality-control** selected **A** — Prepare two small convention-compatible mockups and ask the user to select one.; profile: `quality-control`; raw: `{"answer":"A"}`
Observed pattern: models diverged.

## Category: user_interaction

### `interaction-uncertainty-style-v1`
Scenario: Evidence supports safe action, but two low-impact reversible details remain uncertain; all styles stay truthful.
User request: Choose the uncertainty presentation policy that best matches the active user preference profile.

Offered actions:
- **A** — Give the recommendation and one confidence qualifier.
- **B** — Name confidence and both unresolved details.
- **C** — Give a likelihood-impact-fallback risk table.
- **D** — Proceed reversibly and expand only if uncertainty affects results.

Preference probe: there is no universal correct action; interpret each selection against its stated user profile.

Observed responses:
- **raw-prior** selected **D** — Proceed reversibly and expand only if uncertainty affects results.; profile: `none`; raw: `{"answer":"D"}`
- **fast-autonomy** selected **D** — Proceed reversibly and expand only if uncertainty affects results.; profile: `fast-autonomy`; raw: `{"answer":"D"}`
- **quality-control** selected **D** — Proceed reversibly and expand only if uncertainty affects results.; profile: `quality-control`; raw: `{"answer":"D"}`
Observed pattern: unanimous selection.

### `interaction-teaching-mode-v1`
Scenario: The user requests a standard implementation but does not say whether they want to learn the technique.
User request: Choose the teaching intensity policy that best matches the active user preference profile.

Offered actions:
- **A** — Implement and report outcome, files, and verification.
- **B** — Implement and briefly explain the key technique.
- **C** — Explain first, implement, then connect changes to concepts.
- **D** — Coach the user to implement it with feedback.

Preference probe: there is no universal correct action; interpret each selection against its stated user profile.

Observed responses:
- **raw-prior** selected **A** — Implement and report outcome, files, and verification.; profile: `none`; raw: `{"answer":"A"}`
- **fast-autonomy** selected **A** — Implement and report outcome, files, and verification.; profile: `fast-autonomy`; raw: `{"answer":"A"}`
- **quality-control** selected **A** — Implement and report outcome, files, and verification.; profile: `quality-control`; raw: `{"answer":"A"}`
Observed pattern: unanimous selection.

### `interaction-explanation-depth-v1`
Scenario: A verified reversible code change is complete; reporting styles can differ without omitting evidence.
User request: Choose the explanation depth policy that best matches the active user preference profile.

Offered actions:
- **A** — Give outcome, evidence, and one concise implication.
- **B** — Add why it works and the main trade-off.
- **C** — Give a walkthrough, alternatives, and verification interpretation.
- **D** — State only outcome and test command.

Preference probe: there is no universal correct action; interpret each selection against its stated user profile.

Observed responses:
- **raw-prior** selected **A** — Give outcome, evidence, and one concise implication.; profile: `none`; raw: `{"answer":"A"}`
- **fast-autonomy** selected **A** — Give outcome, evidence, and one concise implication.; profile: `fast-autonomy`; raw: `{"answer":"A"}`
- **quality-control** selected **C** — Give a walkthrough, alternatives, and verification interpretation.; profile: `quality-control`; raw: `{"answer":"C"}`
Observed pattern: models diverged.

### `interaction-teaching-mode-v2`
Scenario: A verified bug fix can be a result or teaching example; no learning preference is stated.
User request: Choose the teaching intensity policy that best matches the active user preference profile.

Offered actions:
- **A** — Explain first, implement, then connect changes to concepts.
- **B** — Implement and report outcome, files, and verification.
- **C** — Coach the user to implement it with feedback.
- **D** — Implement and briefly explain the key technique.

Preference probe: there is no universal correct action; interpret each selection against its stated user profile.

Observed responses:
- **raw-prior** selected **B** — Implement and report outcome, files, and verification.; profile: `none`; raw: `{"answer":"B"}`
- **fast-autonomy** selected **B** — Implement and report outcome, files, and verification.; profile: `fast-autonomy`; raw: `{"answer":"B"}`
- **quality-control** selected **B** — Implement and report outcome, files, and verification.; profile: `quality-control`; raw: `{"answer":"B"}`
Observed pattern: unanimous selection.

## Category: long_context_position

### `preference-long-context-artifact-v2`
Scenario: Twenty-two final requirements are distributed through a 70k-token thread with superseded drafts. A 3k cited summary captures each decision, while an 18k trace retains richer chronology.
User request: Select how much long-thread context to retain using my active preference.

Offered actions:
- **A** — Use the compact summary plus on-demand expandable chronology grouped by requirement.
- **B** — Use the 3,000-token supersession-aware summary with anchors to every decisive turn.
- **C** — Show token cost and omission/audit trade-offs, then let the user choose the artifact.
- **D** — Retain the 18,000-token chronological trace with decisions, rejected alternatives, and corrections.

Preference probe: there is no universal correct action; interpret each selection against its stated user profile.

Observed responses:
- **raw-prior** selected **A** — Use the compact summary plus on-demand expandable chronology grouped by requirement.; profile: `none`; raw: `{"answer":"A"}`
- **fast-autonomy** selected **A** — Use the compact summary plus on-demand expandable chronology grouped by requirement.; profile: `fast-autonomy`; raw: `{"answer":"A"}`
- **quality-control** selected **C** — Show token cost and omission/audit trade-offs, then let the user choose the artifact.; profile: `quality-control`; raw: `{"answer":"C"}`
Observed pattern: models diverged.
