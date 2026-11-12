# Conversation continuity repair plan

> Execute inline using superpowers:executing-plans. Preserve pre-existing user changes.

**Goal:** Make ordinary multi-message local sessions retain decisions, recover older chat,
and show truthful activity and cancellation.

**Approved scope:** User approved continuity, history retrieval and activity/cancellation fixes;
retain the current architecture and interface. Runtime experiments use a temporary directory
under ~/Proyects and the existing Strata service at 127.0.0.1:8080.

**Approach:** Keep recent conversation within a capacity-aware text budget and carry the same
history to execution, including direct routing. Keep older detail searchable through the user's
existing chat_history_search tool. Fix demonstrated activity/cancellation defects at their owners.
No transcript replay of tool output; resumed execution checkpoints remain separately bounded.

## Tasks

- [x] Add failing regressions for decisions older than three exchanges, long user messages,
  deterministic ordering and exclusion of the already-stored current user message.
- [x] Implement bounded conversation context in engine/session_context.py, share it between
  chat_agent.py and pipeline.py; retain the compact fallback for small/unknown contexts.
- [x] Reproduce history retrieval failures: Unicode case matching, long-message anchors,
  semantic coverage, tool availability. Repair only demonstrated failures with regressions.
- [x] Add activity/cancellation regressions; fix classic elapsed time and chat cancellation
  without allowing a cancelled chat turn to escalate into work.
- [x] Drive the actual TUI through several messages, correction, follow-up, cancellation and
  resume against Strata; retain transcripts and a reusable local driver in the test workspace.
- [x] Run focused tests and the complete uv run pytest suite, inspect the diff, review fixes.

## Review focus

- Decisions in early messages survive short acknowledgements and hidden work logs.
- Current input appears exactly once; historical requests do not grant new authority.
- Tool retrieval reports real omissions instead of claiming a truncated prefix is the full row.
- Cancellation prevents later tools/escalation; no overlapping engine work starts.
- Small/unknown windows remain bounded and resumed sessions retain their execution checkpoint.

## Evidence and rulings

Baseline classic session: five messages; Faro correction retained but purpose, SQLite and no-login
were reported missing despite persisted user turns. The model tried chat_history_search with
long AND queries and invalid parameters. The activity timer stayed at 0s throughout chat.
Ruling: use a branch in the current checkout because the user has substantial uncommitted history
and provider changes required by the experiment; a separate checkout would omit those inputs.
No automatic commits of those pre-existing changes.


## Validation ledger

- RED/GREEN: early decisions, long-message middle constraints, current-turn deduplication,
  execution handoff, Unicode case-insensitive search, legacy clipped projections,
  hit-centered excerpts, semantic-vector invalidation, chat cancellation barrier,
  immediate classic result startup, non-execution input queue and synthetic cancel ack.
- Real TUI (prompt_toolkit Application + real input events, no provider mocks): six-message
  chat retained Faro/SQLite/no login/40 min/Wednesday; explicit chat_history_search cited
  the original message. Reopened process retained the same decisions.
- A correction sent while chat was active was queued and processed, and the next turn
  correctly recalled the new 50-minute duration. Cancellation returned in 13.2 s rather
  than the baseline 53.8 s; no synthetic acknowledgement prompt was queued.
- Strata capacity discovery now reads 262144 from its selected /v1/models record.
- Reviewer found missing authenticated route parameters and Unicode casefold excerpt
  offsets; both reproduced with failing tests and fixed. UTF-8 history bounds additionally
  preserve the compact fallback for emoji-heavy input.
- Baseline suite: 5568 passed, 12 skipped, one existing logging-capture failure.
  Isolated that test's logger explicitly instead of changing production's silent logging.
- First full verification: 5587 passed, 12 skipped, 11 existing provider-serializer warnings.
  Final full rerun includes the additional review regressions.

## Remaining observed limitations

- Local provider latency remains variable: resumed answers took ~22 s; one answer after
  cancellation took ~83 s. This pass does not establish immediate network cancellation.
- Classic forced interruption can still leave a provider-blocked daemon worker beyond its
  10 s join timeout (pre-existing); do not describe force cancellation as an immediate
  termination guarantee.
- Semantic transcript retrieval still examines the newest 500 transcript rows and uses the
  embedding model's own text window. Text search covers stored human prose; do not equate
  semantic search with exhaustive historical recall.

- Scope refinement from real work-session evidence: a fully specified Spanish JSON creation
  request entered the multi-pass spec elaborator because .json and imperative "verifica"
  were not recognized. Extend the existing grounded-request detector (not the architecture)
  and cover both concrete follow-ups and an ambiguous JSON request. This preserves the
  elaboration gate for genuinely underspecified work.
- Second complete suite, including review fixes: 5590 passed, 12 skipped, 11 existing
  provider-serializer warnings. Final rerun adds the grounded-JSON routing regressions.

- Grounded-JSON review refinement: dependency/style constraints alone must not skip
  grounding. Added two failing reviewer negatives; new Spanish scope constraints require
  explicit keys or a prior-agreement reference. 91 focused routing tests passed.
- The long work-session controller hit its 600 s deadline after the model created the
  correct JSON, passed strict Python verification and re-read it, while still deciding
  completion. Stopped that timed-out test process. This is a runtime limitation, not a
  passing end-to-end execution claim. Repeating the tiny rename in the same session with
  THINKING_ENABLED=false only in the temporary workspace isolates orchestration from
  local-model reasoning cost. Repository settings are untouched.
- Third complete suite: 5593 passed, 12 skipped, 11 existing serializer warnings.


## Final observed result

- Final full suite: **5595 passed, 12 skipped, 11 existing serializer warnings**
  (`uv run pytest -q`, /tmp/infinidev-ux-final-verified.log).
- `git diff --check` passed. No commits, merges, pushes or release/install changes.
- Real resumed edit completed with a final assistant reply in 441.3 s under the temporary
  THINKING_ENABLED=false setting. The model changed only nombre to Faro Azul and independently
  verified all six JSON fields with Python. A live human-style instruction asked it to stop
  further checking and summarize; the TUI delivered that instruction.
- All owned test drivers have exited. Temporary scripts, screen renders, transcripts and
  ACUERDOS.json are retained at /home/andres/Proyects/infinidev-ux-xc_8j6u0.
- This closes the approved first repair pass; it does not establish that local execution is
  comfortably fast. Strong remaining leads: per-turn task-policy classification/LLM fallback,
  discovery-step selection for natural Spanish changes, and synchronous hidden work-summary
  synthesis before returning the final reply. The task profile for the explicit "cambia" request
  reported answer authority and selected an investigation bootstrap, despite the requested edit.
  Diagnose those together before changing authority heuristics or removing continuity safeguards.

## Subsequent integration

The user subsequently authorized latency fixes and pushing the changes to main. The follow-up
is recorded in 2026-11-12-task-latency.md. Combined final verification: 5616 passed, 12 skipped,
11 existing provider serializer warnings. Include the supporting pre-existing history/provider/
shutdown changes exercised by that suite; personal .agents configuration and runtime evidence
remain outside the commit.
