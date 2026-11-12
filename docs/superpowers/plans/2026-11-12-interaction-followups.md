# Interaction follow-up repairs

User-authorized continuation of the local Strata interaction audit, based on main d7fd30c.
Use a managed worktree, preserve personal .agents configuration and the shared service,
and evaluate tokens and model/tool calls rather than variable wall time.

## Completed implementation

- [x] Replace natural-language regex execution/handoff/promise shortcuts with the primary
  chat model’s native routing and one bounded semantic recovery check for final replies.
  Strict respond/escalate JSON, 256 output tokens, no tools, one attempt, 30s timeout,
  thinking disabled. Invalid output retains the reply; cancellation prevents late handoff.
- [x] Keep the literal user request separate from injected repository/history/hook context.
  Preserve attachments, autonomous hints, streaming state and permission boundaries.
- [x] Explicitly disable local llama.cpp/vLLM template thinking for classification aliases;
  the generic disabled-thinking setting alone did not stop hidden reasoning with Strata.
- [x] Release permission/stdin Event waits if the Escape hold watcher cancels after a modal
  opens. Deny permission rather than allowing the blocked action.
- [x] Meter streamed chat schemas and provider usage-only chunks. Request streaming usage
  only for supported providers; native Ollama receives no unsupported flag.
- [x] Render genuinely empty project directories as ordinary results, retaining permission
  errors. Use conversation/work summaries for prior decisions and files for current claims.
- [x] Give explicit process environment settings priority over saved project settings,
  retaining direct Settings constructor keyword precedence.
- [x] Preserve complete leading lines of explicit oversized file ranges with an accurate
  continuation. Whole-file reads retain the structural outline behavior.
- [x] Browse explicitly requested Git-ignored bases with normal junk filters. Explain
  filtering and include_ignored in local list_directory/glob results.
- [x] Fix a pre-existing surrogate escape in a test docstring that blocked Python3.13
  pytest collection; production Unicode behavior is unchanged.
- [x] Finish full-suite verification.
- [ ] Integrate into main and push.

## Evidence

Focused regressions were demonstrated failing before each repair. Latest checks:

- 174 focused routing/streaming/paging/config/cancellation/discovery/meter tests passed.
- After literal-request and local-template corrections, 116 semantic/chat/streaming/
  pipeline tests passed with Ken sessions disabled.
- Ignored-base discovery and existing listing-budget/read-only checks:17passed.
- Independent reviewer confirmed the literal-request fix, local template controls,
  ranged paging and discovery behavior; no remaining concrete defects reported.
- Full suite: 5685 passed, 13 skipped, 11 existing provider serialization warnings
  in 685.63s (Python3.13); /tmp/infinidev-interaction-semantic-full.log.

Real local Strata checks:

- Six semantic cases classified as expected: explanation, quote, negation, contextual
  approval, reading a file, bug implementation. Total1664input /98generated tokens.
  This is a small probe, not a comprehensive accuracy measurement.
- The first probe returned empty content at256tokens for a file-reading case; the explicit
  local-template override resolved it without increasing the budget.
- Earlier three-turn prompt change A/B:4completions/2reads in both; before42235input/
  880generated, after42134/918. No demonstrated tool-call reduction from the prompt edit.
- TUI context meter matched provider input, including a10810token final stream.

## Real Bruma continuation

Current Bruma changes checkpointed at66a5cc6 before continuing, as requested. No subsequent
Source changes. Runtime .infinidev stays untracked; private .superpowers artifacts preserved.

The first orientation selected historical8a1 because a ranged Continuar.md read was replaced
by an outline that hid the current header. Actual repaired paging smoke retained the8a3b
header in7805characters and pointed to offset27/limit94. A later same-session turn failed
because ignored documents appeared absent; direct reads recovered the evidence. Finally,
the same persisted conversation verified all four time architecture precisions in the three
current drafts. Final Root adoption remains pending; no terminal implementation claimed.

Evidence: /tmp/infinidev-latency-evidence/interaction2_live, interaction2_confirm,
bruma_real, bruma_real_resume; /tmp/infinidev-semantic-local-eval-final.json.

## Remaining observed issues

- The chat’s fixed small read budget interrupts longer read-only investigations.
- Recursive glob enumerates ignored trees before filtering.
- Quit/shutdown worker release and pending-task restart need a separate lifecycle fix.
- Final developer review can consume substantially more tokens than the implementation.

These are not fixed by this batch. Keep subsequent changes grounded in real sessions.
