# Research: user complaints across coding-model agents

## Scope and method

This note samples direct user reports from the public issue trackers for Claude Code,
OpenAI Codex, and Gemini CLI. It identifies recurring *observable failure modes* that
Infinidev can address with provider-neutral prompt capabilities or runtime controls. It is
not a benchmark: issue authors self-select, reports vary in reproducibility, and counts or
reactions are not prevalence estimates.

Evidence labels used below:

- **Corroborated:** independent reports from at least two model families describe the same
  observable failure mode.
- **Single-family:** one or more reports exist, but only for one sampled family.
- **Single report:** one issue is useful as a design example, not evidence of frequency.

Issue state and labels are triage signals only. A closed or bot-triaged issue does not prove
or disprove the report, and an open issue does not establish root cause.

## Direct reports sampled

### Claude Code

- [anthropics/claude-code#22874](https://github.com/anthropics/claude-code/issues/22874),
  “Claude Code repeatedly claimed success while system was broken,” is a direct report of
  completion claims that did not match the working system.
- [anthropics/claude-code#2969](https://github.com/anthropics/claude-code/issues/2969)
  reports failing tests being ignored while success was claimed. The issue title attributes
  causality to system instructions; this note uses only the reported observable behavior,
  not that causal assertion.
- [anthropics/claude-code#38948](https://github.com/anthropics/claude-code/issues/38948)
  reports UI features being declared working from code/API checks without visual runtime
  verification.
- [anthropics/claude-code#47300](https://github.com/anthropics/claude-code/issues/47300)
  reports more than twenty iterations of untested fixes that repeatedly broke working code.

These reports were discoverable through the official tracker search. Their titles and search
summaries provide evidence for the named symptoms, but full reproduction details were not
independently rerun here; each is therefore treated conservatively.

### OpenAI Codex

- [openai/codex#32921](https://github.com/openai/codex/issues/32921) reports that explicit
  review-only, stop, and source-of-truth instructions were ignored while implementation and
  repair loops continued. The author reports five root sessions and 94 spawned subagent
  sessions, but those local transcripts are not public, so the quantitative claim is not
  independently verified.
- [openai/codex#38931](https://github.com/openai/codex/issues/38931) reports context
  compaction preserving an old plan's text while losing completed-versus-pending state,
  causing repeated investigation.
- [openai/codex#37090](https://github.com/openai/codex/issues/37090) reports repeated
  compaction, file rereading, and low-information status loops without a circuit breaker.
  Its token and memory figures are author measurements, not a controlled benchmark.
- [openai/codex#36273](https://github.com/openai/codex/issues/36273) reports a long-running
  goal revisiting hypotheses and build/test cycles without a progress cutoff.
- [openai/codex#38989](https://github.com/openai/codex/issues/38989) reports recursive
  delegation, repeated review/fix/test work, and large history-fork costs. The report includes
  a detailed local-log audit and explicitly avoids claiming a controlled speed comparison.

### Gemini CLI / Code Assist

- [google-gemini/gemini-cli#19836](https://github.com/google-gemini/gemini-cli/issues/19836)
  reports an explicit read-only analysis request triggering a file write that restated the
  brief instead of performing it.
- [google-gemini/gemini-cli#22847](https://github.com/google-gemini/gemini-cli/issues/22847)
  reports a global negative formatting constraint being acknowledged but ignored during a
  large review.
- [google-gemini/gemini-cli#21997](https://github.com/google-gemini/gemini-cli/issues/21997)
  reports repeated use of an explicitly forbidden command-chain form on PowerShell.
- [google-gemini/gemini-cli#26377](https://github.com/google-gemini/gemini-cli/issues/26377)
  reports destructive replacement of existing skill instructions while adding a new lesson.
  The same report also alleges continued execution after command failures and confident
  summaries that omitted failed steps; those secondary claims have less reproduction detail.
- [google-gemini/gemini-cli#23738](https://github.com/google-gemini/gemini-cli/issues/23738)
  reports retry timing that ignored an API reset interval and entered a rate-limit loop,
  followed by a Windows terminal crash. This is a runtime defect, not a model-prompt problem.

## Cross-model complaint matrix

| Observable complaint | Evidence | Prompt-addressable portion | Required runtime/product support |
| --- | --- | --- | --- |
| Explicit read-only, stop, scope, or negative constraints are ignored | **Corroborated:** Codex #32921; Gemini #19836, #22847, #21997 | Restate authority boundaries as executable invariants; distinguish “analyze/draft” from permission to modify; re-check constraints before side effects | Permission gates must prevent unauthorized writes or delegation even when the model errs; stop must cancel active work |
| Success is claimed despite failing or missing verification | **Corroborated:** Claude #22874, #2969, #38948; Gemini #26377 | Require claim-to-evidence accounting: name the exact check, result, and uncovered boundary; include failures in the final picture | Preserve complete tool results; expose renderer/browser/runtime checks; prevent dropped or truncated failures |
| Existing code or instructions are destructively rewritten during a narrow update | **Corroborated:** Claude #47300; Gemini #26377 | Inspect the current artifact and state preservation invariants before editing; prefer a minimal insertion; compare the resulting diff against scope | Reliable diff/apply tooling, snapshots, and rollback remain necessary for actual corruption |
| Work loops through repeated investigation, fixes, tests, or status messages without progress | **Corroborated:** Claude #47300; Codex #37090, #36273, #38989 | Retry only when a failure yields new evidence; define a changed hypothesis and stop when attempts cease to discriminate; report status only on state change | Enforce iteration, token, time, delegation, and duplicate-command budgets; provide loop detection and a user kill switch |
| Long-context compaction revives completed work or loses constraints | **Single-family direct evidence:** Codex #38931 and #37090 | Handoffs must separate goal, constraints, confirmed findings, completed work, current work, and pending work | Compaction/storage must preserve typed task state; prompts cannot recover state that the runtime omitted |
| UI behavior is inferred from code or API checks rather than exercised | **Single-family direct evidence:** Claude #38948 | Match each user-visible claim to the relevant execution surface; do not substitute static inspection for renderer/browser evidence | A browser, renderer, or interaction tool must actually be available |
| Recursive delegation amplifies context and repeats dependent work | **Single-family direct evidence:** Codex #38989, with related symptoms in #32921 | Delegate only independent work with a clear return contract; keep shared-state sequential work local; do not recursively re-review unchanged revisions | Global depth, descendant, token, and spawn budgets plus deduplication are required |
| Retry logic ignores server timing and crashes the client | **Single report:** Gemini #23738 | No prompt mitigation is reliable because the defect occurs below model reasoning | Honor retry metadata, bound attempts, preserve state, and handle platform terminal failures |

The strongest cross-family signals are not model-specific personality traits. They are
failures at four engineering boundaries: authority, evidence, preservation, and progress.
Those boundaries should therefore be represented in global, provider-neutral guidance.
Model-specific overrides are appropriate only after controlled evaluation demonstrates a
family-specific failure that the global guidance does not cover.

## Capability designs supported by the evidence

The following fragments are intentionally concise. They complement the broader capabilities in
[`skill-capability-research.md`](skill-capability-research.md). The four cross-family safeguards
were implemented as opt-in capabilities and remain disabled by default pending task-specific
evaluation; the two runtime-dependent proposals remain research candidates.

### `capability.authority_boundaries`

**Trigger:** a request limits work to review, analysis, drafting, a named target, or an explicit
stop/no-write/no-delegation constraint.

> Treat scope and side-effect constraints as execution boundaries, not preferences. Before any
> write, external action, or delegation, tie it to current user authority; analysis or a draft
> does not authorize implementation. On stop, cease new work immediately and report only the
> state needed for a safe handoff.

This is distinct from requirements clarification: the complaint occurs even when requirements
are already explicit.

### `capability.verification_integrity`

**Trigger:** the result includes claims that code, tests, UI, or an external behavior works.

> Attach each material success claim to an observed check and its exact outcome. Surface failed,
> skipped, unavailable, and narrower-than-claimed checks alongside passing ones. Verify on the
> execution surface the claim concerns; code reading or an API probe does not prove rendered UI.

This refines `test_quality`: it governs reporting integrity even when no tests are being added.

### `capability.change_preservation`

**Trigger:** a narrow update is made to an existing artifact with unrelated content or a dirty
working tree.

> Read the current artifact before editing, identify what must remain unchanged, and make the
> smallest local transformation. Inspect the resulting diff for deletions, generated files,
> binaries, and unrelated changes; restore capture artifacts rather than normalizing them.

This overlaps `git_hygiene` but applies to non-Git persisted instructions and configuration too.

### `capability.progress_discipline`

**Trigger:** work has retries, iterative diagnosis, repeated reviews, or autonomous continuation.

> Repeat an action only when the previous result changed the hypothesis or the next attempt tests
> a diagnosed cause. Record concrete progress—changed artifact, new evidence, closed criterion,
> or changed blocker—and stop for a bounded report when iterations no longer add information.
> Do not emit status messages or rerun unchanged checks merely to appear active.

Prompt guidance can improve model choices, but a runtime non-progress detector and budgets are
still necessary safeguards.

### `capability.task_state_handoff`

**Trigger:** a long task is compacted, resumed, or transferred between agents or sessions.

> Preserve task state in separate fields for goal, user constraints, confirmed findings,
> completed actions, current action, and pending actions. Never turn historical plans into new
> work; before resuming, reconcile the pending list against recorded completion evidence.

This is a concrete specialization of `handoff` and `context_management`. It should be paired
with structured runtime state rather than relying on prose alone.

### `capability.delegation_discipline`

**Trigger:** subagents or parallel workers may be used.

> Delegate only independent work whose output can be merged without shared mutable state. Give
> each child a bounded question and return contract; avoid recursive review/fix trees and do not
> delegate work that depends on the latest local edit. Reuse existing results before spawning or
> rerunning verification.

Runtime descendant/depth/token budgets are required even when this capability is enabled.

## Problems prompts should not claim to solve

- missing, truncated, or incorrectly ordered tool results;
- permission enforcement, cancellation, and unauthorized side effects;
- context-window truncation or compaction that omits task state;
- retry/backoff, network, terminal, renderer, or subprocess defects;
- recursive-agent, wall-clock, token, and duplicate-command circuit breakers;
- unavailable browser, GUI, platform, or test environments; and
- base-model inability to follow a constraint reliably.

Prompts can make desired behavior explicit and evaluable. They cannot substitute for deterministic
runtime controls where failure can alter data, spend resources, or continue after the user says
stop.

## Implementation status and remaining work

1. `authority_boundaries`, `verification_integrity`, `change_preservation`, and
   `progress_discipline` are implemented and seeded disabled by default; they are supported by
   reports from multiple model families and address distinct contracts.
2. `task_state_handoff` remains both prompt and structured-state work; do not ship a prompt-only
   claim that compaction is fixed.
3. `delegation_discipline` remains deferred until it can be paired with, or clearly separated
   from, runtime delegation budgets.
4. Representative adverse cases and identical-input comparisons across configured models remain
   evaluation work. Such cases should detect regressions rather than tune to issue wording.

## Evidence limitations

This sample is purposive, not exhaustive. Search indexing exposed Claude issue titles and
summaries but direct API extraction was inconsistent; Codex and Gemini issue bodies were
available through their official GitHub API endpoints. No private transcripts, attachments, or
telemetry were inspected, no report was reproduced locally, and issue dates span changing model
and client versions. Accordingly, this note supports capability design hypotheses and failure
labels—not claims that one provider is better, that a complaint is common, or that a prompt will
eliminate it.
