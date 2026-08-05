# Choice-only versus self-report elicitation

Decision criteria and confidence are model self-reports, not privileged access to private reasoning. Because elicitation itself can change a choice, choice-only and self-report observations remain separate evidence conditions.

All calls were isolated, had no system message, and used no user preference profile.

## Summary
- **Sol**: 5/10 choices changed; median self-reported confidence 0.91.
- **Terra**: 3/10 choices changed; median self-reported confidence 0.84.
- **Luna**: 3/10 choices changed; median self-reported confidence 0.90.

## Category: decision_support_for_users

### `decision-follow-up-cadence-v2`

Scenario: A trial can be reconsidered at several natural checkpoints, and no safety threshold mandates one particular review schedule.
User request: Choose the decision review cadence policy that best matches the active user preference profile.

- **Sol** (unchanged)
  - Choice-only: **A** — Let the user choose the review schedule from expected information gain and cost.
  - Self-report: **A** — Let the user choose the review schedule from expected information gain and cost.
  - Expressed criterion: When no checkpoint is safety-mandated, let the user select the cadence based on expected information gain and review cost.
  - Stated missing context: The user's specific preferences regarding review burden, responsiveness, and cost are not provided.
  - Verbal confidence: 0.92
  - Raw self-report: `{"answer":"A","confidence":0.92,"decision_criterion":"When no checkpoint is safety-mandated, let the user select the cadence based on expected information gain and review cost.","missing_context":"The user's specific preferences regarding review burden, responsiveness, and cost are not provided."}`

- **Terra** (unchanged)
  - Choice-only: **A** — Let the user choose the review schedule from expected information gain and cost.
  - Self-report: **A** — Let the user choose the review schedule from expected information gain and cost.
  - Expressed criterion: No mandatory safety schedule applies, so cadence should be selected based on user preferences, expected information gain, and review cost.
  - Stated missing context: Active user preference profile details are not provided.
  - Verbal confidence: 0.9
  - Raw self-report: `{"answer":"A","confidence":0.9,"decision_criterion":"No mandatory safety schedule applies, so cadence should be selected based on user preferences, expected information gain, and review cost.","missing_context":"Active user preference profile details are not provided."}`

- **Luna** (unchanged)
  - Choice-only: **A** — Let the user choose the review schedule from expected information gain and cost.
  - Self-report: **A** — Let the user choose the review schedule from expected information gain and cost.
  - Expressed criterion: When no safety constraint fixes the cadence, let the user select the schedule by balancing expected information gain against review cost.
  - Stated missing context: (none)
  - Verbal confidence: 0.93
  - Raw self-report: `{"answer":"A","confidence":0.93,"decision_criterion":"When no safety constraint fixes the cadence, let the user select the schedule by balancing expected information gain against review cost.","missing_context":""}`

### `decision-stakeholder-input-v1`

Scenario: A reversible team workflow choice affects three stakeholders differently; the requesting user has authority to decide alone.
User request: Choose the stakeholder participation policy that best matches the active user preference profile.

- **Sol** (changed)
  - Choice-only: **A** — Help the authorized user decide now from the known impacts.
  - Self-report: **D** — Provide a provisional reversible choice plus a feedback window.
  - Expressed criterion: Use reversibility to preserve the authorized user’s agency while creating a structured opportunity for affected stakeholders to provide feedback.
  - Stated missing context: (none)
  - Verbal confidence: 0.84
  - Raw self-report: `{"answer":"D","confidence":0.84,"decision_criterion":"Use reversibility to preserve the authorized user’s agency while creating a structured opportunity for affected stakeholders to provide feedback.","missing_context":""}`

- **Terra** (unchanged)
  - Choice-only: **A** — Help the authorized user decide now from the known impacts.
  - Self-report: **A** — Help the authorized user decide now from the known impacts.
  - Expressed criterion: The authorized user can make a reversible decision directly using known stakeholder impacts.
  - Stated missing context: The active user preference profile, especially their preference for stakeholder consultation versus speed and autonomy, is not specified.
  - Verbal confidence: 0.52
  - Raw self-report: `{"answer":"A","confidence":0.52,"decision_criterion":"The authorized user can make a reversible decision directly using known stakeholder impacts.","missing_context":"The active user preference profile, especially their preference for stakeholder consultation versus speed and autonomy, is not specified."}`

- **Luna** (unchanged)
  - Choice-only: **A** — Help the authorized user decide now from the known impacts.
  - Self-report: **A** — Help the authorized user decide now from the known impacts.
  - Expressed criterion: The user has authority to decide, and the active preference profile supports choosing directly from the known stakeholder impacts.
  - Stated missing context: (none)
  - Verbal confidence: 0.97
  - Raw self-report: `{"answer":"A","confidence":0.97,"decision_criterion":"The user has authority to decide, and the active preference profile supports choosing directly from the known stakeholder impacts.","missing_context":""}`

## Category: user_interaction

### `interaction-explanation-depth-v1`

Scenario: A verified reversible code change is complete; reporting styles can differ without omitting evidence.
User request: Choose the explanation depth policy that best matches the active user preference profile.

- **Sol** (changed)
  - Choice-only: **A** — Give outcome, evidence, and one concise implication.
  - Self-report: **B** — Add why it works and the main trade-off.
  - Expressed criterion: Use moderate detail: report the verified outcome and evidence, then briefly explain why it works and the main trade-off.
  - Stated missing context: (none)
  - Verbal confidence: 0.86
  - Raw self-report: `{"answer":"B","confidence":0.86,"decision_criterion":"Use moderate detail: report the verified outcome and evidence, then briefly explain why it works and the main trade-off.","missing_context":""}`

- **Terra** (unchanged)
  - Choice-only: **A** — Give outcome, evidence, and one concise implication.
  - Self-report: **A** — Give outcome, evidence, and one concise implication.
  - Expressed criterion: Use concise reporting that includes the verified outcome, supporting evidence, and a practical implication.
  - Stated missing context: No explicit user preference profile beyond the request context.
  - Verbal confidence: 0.86
  - Raw self-report: `{"answer":"A","confidence":0.86,"decision_criterion":"Use concise reporting that includes the verified outcome, supporting evidence, and a practical implication.","missing_context":"No explicit user preference profile beyond the request context."}`

- **Luna** (unchanged)
  - Choice-only: **A** — Give outcome, evidence, and one concise implication.
  - Self-report: **A** — Give outcome, evidence, and one concise implication.
  - Expressed criterion: Use a concise report that includes the outcome, verification evidence, and one practical implication.
  - Stated missing context: The active user preference profile is not explicitly provided.
  - Verbal confidence: 0.72
  - Raw self-report: `{"answer":"A","confidence":0.72,"decision_criterion":"Use a concise report that includes the outcome, verification evidence, and one practical implication.","missing_context":"The active user preference profile is not explicitly provided."}`

### `interaction-teaching-mode-v1`

Scenario: The user requests a standard implementation but does not say whether they want to learn the technique.
User request: Choose the teaching intensity policy that best matches the active user preference profile.

- **Sol** (changed)
  - Choice-only: **A** — Implement and report outcome, files, and verification.
  - Self-report: **B** — Implement and briefly explain the key technique.
  - Expressed criterion: Default to implementation with a concise explanation when no explicit learning preference is stated.
  - Stated missing context: Whether the user prefers outcome-only delivery or a more instructional approach.
  - Verbal confidence: 0.8
  - Raw self-report: `{"answer":"B","confidence":0.8,"decision_criterion":"Default to implementation with a concise explanation when no explicit learning preference is stated.","missing_context":"Whether the user prefers outcome-only delivery or a more instructional approach."}`

- **Terra** (unchanged)
  - Choice-only: **A** — Implement and report outcome, files, and verification.
  - Self-report: **A** — Implement and report outcome, files, and verification.
  - Expressed criterion: No learning preference was stated, so use the standard direct implementation workflow with outcome, files, and verification.
  - Stated missing context: The active user preference profile is not explicitly provided.
  - Verbal confidence: 0.93
  - Raw self-report: `{"answer":"A","confidence":0.93,"decision_criterion":"No learning preference was stated, so use the standard direct implementation workflow with outcome, files, and verification.","missing_context":"The active user preference profile is not explicitly provided."}`

- **Luna** (unchanged)
  - Choice-only: **A** — Implement and report outcome, files, and verification.
  - Self-report: **A** — Implement and report outcome, files, and verification.
  - Expressed criterion: With no stated preference for learning, use the standard implementation workflow and report the result, changed files, and verification.
  - Stated missing context: (none)
  - Verbal confidence: 0.8
  - Raw self-report: `{"answer":"A","confidence":0.8,"decision_criterion":"With no stated preference for learning, use the standard implementation workflow and report the result, changed files, and verification.","missing_context":""}`

### `interaction-teaching-mode-v2`

Scenario: A verified bug fix can be a result or teaching example; no learning preference is stated.
User request: Choose the teaching intensity policy that best matches the active user preference profile.

- **Sol** (unchanged)
  - Choice-only: **B** — Implement and report outcome, files, and verification.
  - Self-report: **B** — Implement and report outcome, files, and verification.
  - Expressed criterion: With no stated learning preference, prioritize completing the verified fix and concisely reporting results and verification.
  - Stated missing context: (none)
  - Verbal confidence: 0.95
  - Raw self-report: `{"answer":"B","confidence":0.95,"decision_criterion":"With no stated learning preference, prioritize completing the verified fix and concisely reporting results and verification.","missing_context":""}`

- **Terra** (unchanged)
  - Choice-only: **B** — Implement and report outcome, files, and verification.
  - Self-report: **B** — Implement and report outcome, files, and verification.
  - Expressed criterion: No learning preference is stated, so prioritize delivering the verified fix with outcome, changed files, and verification details.
  - Stated missing context: Active user preference profile details are not provided.
  - Verbal confidence: 0.9
  - Raw self-report: `{"answer":"B","confidence":0.9,"decision_criterion":"No learning preference is stated, so prioritize delivering the verified fix with outcome, changed files, and verification details.","missing_context":"Active user preference profile details are not provided."}`

- **Luna** (unchanged)
  - Choice-only: **B** — Implement and report outcome, files, and verification.
  - Self-report: **B** — Implement and report outcome, files, and verification.
  - Expressed criterion: No learning preference is stated, so prioritize direct implementation with a concise outcome and verification report.
  - Stated missing context: (none)
  - Verbal confidence: 0.93
  - Raw self-report: `{"answer":"B","confidence":0.93,"decision_criterion":"No learning preference is stated, so prioritize direct implementation with a concise outcome and verification report.","missing_context":""}`

### `interaction-uncertainty-style-v1`

Scenario: Evidence supports safe action, but two low-impact reversible details remain uncertain; all styles stay truthful.
User request: Choose the uncertainty presentation policy that best matches the active user preference profile.

- **Sol** (unchanged)
  - Choice-only: **D** — Proceed reversibly and expand only if uncertainty affects results.
  - Self-report: **D** — Proceed reversibly and expand only if uncertainty affects results.
  - Expressed criterion: Low-impact, reversible uncertainty favors action with conditional elaboration only if outcomes are affected.
  - Stated missing context: (none)
  - Verbal confidence: 0.93
  - Raw self-report: `{"answer":"D","confidence":0.93,"decision_criterion":"Low-impact, reversible uncertainty favors action with conditional elaboration only if outcomes are affected.","missing_context":""}`

- **Terra** (unchanged)
  - Choice-only: **D** — Proceed reversibly and expand only if uncertainty affects results.
  - Self-report: **D** — Proceed reversibly and expand only if uncertainty affects results.
  - Expressed criterion: Safe reversible action is supported; expand uncertainty discussion only if it could affect outcomes.
  - Stated missing context: The active user preference profile is not specified.
  - Verbal confidence: 0.62
  - Raw self-report: `{"answer":"D","confidence":0.62,"decision_criterion":"Safe reversible action is supported; expand uncertainty discussion only if it could affect outcomes.","missing_context":"The active user preference profile is not specified."}`

- **Luna** (unchanged)
  - Choice-only: **D** — Proceed reversibly and expand only if uncertainty affects results.
  - Self-report: **D** — Proceed reversibly and expand only if uncertainty affects results.
  - Expressed criterion: Take the safe reversible action and disclose additional uncertainty only if it could affect the result.
  - Stated missing context: The active user preference profile is not explicitly provided.
  - Verbal confidence: 0.82
  - Raw self-report: `{"answer":"D","confidence":0.82,"decision_criterion":"Take the safe reversible action and disclose additional uncertainty only if it could affect the result.","missing_context":"The active user preference profile is not explicitly provided."}`

## Category: ambiguity_and_clarification

### `preference-ambiguity-batching-v2`

Scenario: Four unrelated UI presentation choices remain open. They do not alter architecture or safety, but each must be decided before implementation is complete.
User request: Select the clarification cadence that matches my active preference.

- **Sol** (changed)
  - Choice-only: **A** — Propose local-convention defaults for all four and ask for a single confirmation or corrections.
  - Self-report: **B** — Ask all four concise questions in one structured message with recommended defaults.
  - Expressed criterion: Bundle independent, low-risk clarification questions to minimize interaction overhead while providing actionable defaults.
  - Stated missing context: No explicit user preference for clarification cadence is provided.
  - Verbal confidence: 0.72
  - Raw self-report: `{"answer":"B","confidence":0.72,"decision_criterion":"Bundle independent, low-risk clarification questions to minimize interaction overhead while providing actionable defaults.","missing_context":"No explicit user preference for clarification cadence is provided."}`

- **Terra** (changed)
  - Choice-only: **A** — Propose local-convention defaults for all four and ask for a single confirmation or corrections.
  - Self-report: **B** — Ask all four concise questions in one structured message with recommended defaults.
  - Expressed criterion: Batch independent low-risk decisions in one concise, structured clarification while providing recommended defaults.
  - Stated missing context: The user's stated preference for default-first confirmation versus explicit per-choice questions is not provided.
  - Verbal confidence: 0.78
  - Raw self-report: `{"answer":"B","confidence":0.78,"decision_criterion":"Batch independent low-risk decisions in one concise, structured clarification while providing recommended defaults.","missing_context":"The user's stated preference for default-first confirmation versus explicit per-choice questions is not provided."}`

- **Luna** (changed)
  - Choice-only: **D** — Ask one question at a time and use each answer to frame the next interaction.
  - Self-report: **B** — Ask all four concise questions in one structured message with recommended defaults.
  - Expressed criterion: Four unrelated choices are best handled in one concise, structured clarification round with recommended defaults.
  - Stated missing context: The user's explicit active clarification preference is not stated.
  - Verbal confidence: 0.72
  - Raw self-report: `{"answer":"B","confidence":0.72,"decision_criterion":"Four unrelated choices are best handled in one concise, structured clarification round with recommended defaults.","missing_context":"The user's explicit active clarification preference is not stated."}`

### `preference-ambiguity-timing-v2`

Scenario: Two repository conventions support different names for a private helper. Either choice passes tests, changes no public API, and is cheaply reversible.
User request: Select whether to clarify or infer the name using my active preference.

- **Sol** (unchanged)
  - Choice-only: **D** — Choose the convention used by the nearest analogous helper, implement it, and disclose the assumption.
  - Self-report: **D** — Choose the convention used by the nearest analogous helper, implement it, and disclose the assumption.
  - Expressed criterion: For low-risk, reversible naming choices, follow the nearest analogous local precedent, implement directly, and disclose the assumption.
  - Stated missing context: (none)
  - Verbal confidence: 0.9
  - Raw self-report: `{"answer":"D","confidence":0.9,"decision_criterion":"For low-risk, reversible naming choices, follow the nearest analogous local precedent, implement directly, and disclose the assumption.","missing_context":""}`

- **Terra** (changed)
  - Choice-only: **C** — Use the locally dominant name in a small draft diff and invite correction before broader work.
  - Self-report: **B** — Ask one focused naming question and wait before editing.
  - Expressed criterion: No active preference is provided, so clarify the requested decision rule before making the naming choice.
  - Stated missing context: The user's active preference for clarification versus inference is not specified.
  - Verbal confidence: 0.84
  - Raw self-report: `{"answer":"B","confidence":0.84,"decision_criterion":"No active preference is provided, so clarify the requested decision rule before making the naming choice.","missing_context":"The user's active preference for clarification versus inference is not specified."}`

- **Luna** (changed)
  - Choice-only: **D** — Choose the convention used by the nearest analogous helper, implement it, and disclose the assumption.
  - Self-report: **C** — Use the locally dominant name in a small draft diff and invite correction before broader work.
  - Expressed criterion: Use the locally dominant convention in a small, reversible draft and invite correction before expanding the change.
  - Stated missing context: (none)
  - Verbal confidence: 0.9
  - Raw self-report: `{"answer":"C","confidence":0.9,"decision_criterion":"Use the locally dominant convention in a small, reversible draft and invite correction before expanding the change.","missing_context":""}`

## Category: long_context_position

### `preference-long-context-artifact-v2`

Scenario: Twenty-two final requirements are distributed through a 70k-token thread with superseded drafts. A 3k cited summary captures each decision, while an 18k trace retains richer chronology.
User request: Select how much long-thread context to retain using my active preference.

- **Sol** (changed)
  - Choice-only: **B** — Use the 3,000-token supersession-aware summary with anchors to every decisive turn.
  - Self-report: **C** — Show token cost and omission/audit trade-offs, then let the user choose the artifact.
  - Expressed criterion: When no active retention preference is available, present token-cost and auditability trade-offs and ask the user to choose.
  - Stated missing context: The user's active preference is not provided.
  - Verbal confidence: 0.93
  - Raw self-report: `{"answer":"C","confidence":0.93,"decision_criterion":"When no active retention preference is available, present token-cost and auditability trade-offs and ask the user to choose.","missing_context":"The user's active preference is not provided."}`

- **Terra** (changed)
  - Choice-only: **A** — Use the compact summary plus on-demand expandable chronology grouped by requirement.
  - Self-report: **C** — Show token cost and omission/audit trade-offs, then let the user choose the artifact.
  - Expressed criterion: Active retention preference is not specified, so present cost, completeness, and auditability trade-offs before selecting a context artifact.
  - Stated missing context: The user's active preference for compactness, auditability, or chronology is not provided.
  - Verbal confidence: 0.82
  - Raw self-report: `{"answer":"C","confidence":0.82,"decision_criterion":"Active retention preference is not specified, so present cost, completeness, and auditability trade-offs before selecting a context artifact.","missing_context":"The user's active preference for compactness, auditability, or chronology is not provided."}`

- **Luna** (changed)
  - Choice-only: **A** — Use the compact summary plus on-demand expandable chronology grouped by requirement.
  - Self-report: **C** — Show token cost and omission/audit trade-offs, then let the user choose the artifact.
  - Expressed criterion: The active preference is not specified, so present token-cost and audit trade-offs and let the user choose.
  - Stated missing context: The user's active context-retention preference is not provided.
  - Verbal confidence: 0.93
  - Raw self-report: `{"answer":"C","confidence":0.93,"decision_criterion":"The active preference is not specified, so present token-cost and audit trade-offs and let the user choose.","missing_context":"The user's active context-retention preference is not provided."}`

## Category: vague_requirement_analysis

### `preference-vague-inference-v2`

Scenario: A reversible internal-only diagnostics UI should be made 'cleaner.' Nearby panels establish one strong visual convention; functionality and public behavior remain fixed.
User request: Select whether to infer or ask using my active collaboration preference.

- **Sol** (unchanged)
  - Choice-only: **B** — Apply the nearest established visual convention, preserve behavior, and show the resulting diff.
  - Self-report: **B** — Apply the nearest established visual convention, preserve behavior, and show the resulting diff.
  - Expressed criterion: Infer and act when the change is reversible, low-risk, and governed by a strong nearby convention.
  - Stated missing context: (none)
  - Verbal confidence: 0.95
  - Raw self-report: `{"answer":"B","confidence":0.95,"decision_criterion":"Infer and act when the change is reversible, low-risk, and governed by a strong nearby convention.","missing_context":""}`

- **Terra** (unchanged)
  - Choice-only: **B** — Apply the nearest established visual convention, preserve behavior, and show the resulting diff.
  - Self-report: **B** — Apply the nearest established visual convention, preserve behavior, and show the resulting diff.
  - Expressed criterion: Strong nearby conventions and reversible internal-only scope support inferring the established pattern while preserving behavior and presenting the diff.
  - Stated missing context: The user's active collaboration preference is not explicitly provided.
  - Verbal confidence: 0.84
  - Raw self-report: `{"answer":"B","confidence":0.84,"decision_criterion":"Strong nearby conventions and reversible internal-only scope support inferring the established pattern while preserving behavior and presenting the diff.","missing_context":"The user's active collaboration preference is not explicitly provided."}`

- **Luna** (unchanged)
  - Choice-only: **C** — Make the smallest convention-aligned cleanup first and invite expansion after feedback.
  - Self-report: **C** — Make the smallest convention-aligned cleanup first and invite expansion after feedback.
  - Expressed criterion: Infer the established convention, make the smallest reversible behavior-preserving cleanup, and invite feedback before expanding.
  - Stated missing context: (none)
  - Verbal confidence: 0.9
  - Raw self-report: `{"answer":"C","confidence":0.9,"decision_criterion":"Infer the established convention, make the smallest reversible behavior-preserving cleanup, and invite feedback before expanding.","missing_context":""}`
