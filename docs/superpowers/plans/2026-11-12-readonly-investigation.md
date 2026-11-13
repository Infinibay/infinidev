# Read-only investigation and pruned discovery

Goal: finish requested document/code investigations without a new human prompt, and
avoid entering ignored directories just to discard their files later.

Approach: retain the short initial chat budget. Expose continue_investigation(reason)
as a read-only control tool. Grant five additional iterations only after fresh successful
reads, up to CHAT_AGENT_INVESTIGATION_MAX_ITERATIONS (default20); explicit caller max_iterations remains hard.
Simple replies need no extension or extra classification. Exhausted extended research
reports the incomplete investigation instead of selecting a developer mode for exhaustion.
A blanket larger budget delays simple chats; semantic preclassification adds a model call
before any useful evidence. Neither is needed for this repair.

Glob uses pattern-component traversal and prunes Git/fallback-ignored directories before
descent, honoring include_ignored, explicit bases, max_depth and symlink semantics.

- [x] RED/GREEN chat continuation, fresh-evidence guard, malformed requests, explicit
  hard limits and total ceiling. Files: chat_agent.py, settings.py, chat_agent tool package,
  tool registry and prompt. Tests: test_chat_agent.py plus role/profile regressions.
- [x] RED/GREEN glob traversal spies with real Git nested ignore/negation, fallback,
  depth, include_ignored and matching cases. Files: glob_tool.py, focused new tests.
- [x] Independent review, local Strata multi-message probe, full uv run pytest.
- [ ] Commit, fast-forward main, verify and push. Preserve Bruma and shared Strata state.

## Verification ledger

- Initial chat regressions: 3 failed before implementation, then passed. Broader focused
  chat/streaming/role/config check: 152 passed; final chat regressions: 78 passed.
- Combined final chat/glob/read-only checks: 149 passed. Prompt/profile checks:712passed.
- Traversal spies failed against baseline before repair. Real Git tests preserve both
  direct and nested tracked descendants while refusing to enter generated sibling trees.
- Review found terminal ** differs on Python3.12 versus3.13. Three compatibility cases
  failed on3.12 before the fix; new glob file now passes18tests on each interpreter.
- Independent review reported no remaining concrete findings after that correction.
- Real local Strata, one persisted TUI conversation in an isolated temporary workspace:
  first chain read all8notes and summed460. Second chain required sequential reads:
  two continue_investigation calls,8reads,12model completions including validation,
  126586input/1449generated tokens; completed all8values and sum98 without a new message.
  Follow-up answered remaining sum72 from history with0reads,2completions and
  12456input/258generated. This demonstrates continuity, not a token-reduction claim.
- Full uv run pytest:5712passed,13skipped,11existing provider serialization warnings
  in719.48s (Python3.13); /tmp/infinidev-readonly-full.log.

Real evidence: /tmp/infinidev-latency-evidence/readonly_chain_fixed and
/tmp/infinidev-readonly-metrics.json. Strata unchanged; Bruma source untouched.
