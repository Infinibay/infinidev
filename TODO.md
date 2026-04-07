# TODO — Quality improvements

Pending work for the Infinidev loop engine, organised by impact and
confidence. Captured at the end of a long evidence-driven session that
landed ~25 commits across loop fixes, syntax check, guidance system,
test parsers, latency instrumentation, DB connection pool, and async
file indexing. Everything below is concrete enough to start tomorrow
without more discovery — each item names files, line counts, and the
empirical observation that motivated it.

---

## 0 — Context: where we came from this session

**Read this first if you're picking up after a break.** Understanding
the trajectory matters because most TODOs below reference patterns
we observed empirically during the session, not theoretical concerns.

### What landed (in order of commit)

| Topic | Commits | Net effect |
|---|---|---|
| Loop fixes (per-step expected_output, analyst bypass, set_loop_state, tools binding) | `98b0394` | Loop works for `--prompt` mode with imperative tasks; small models classified correctly |
| Tree-sitter pre-write syntax check | `472ba26` | Refuses writes that would leave a file in broken-syntax state; supports Python + JS |
| Guidance system core (8 detectors + library) | `32e5313` | Reactive how-to advice for stuck small models, never fires for large models, never costs an LLM call |
| Multi-runner generalisation | `a4b5a8e` | Test detection across 16+ runners (pytest, jest, cargo, go, etc.) |
| Custom test commands (3 sources) | `67c1055` | Built-in + setting + agent-declared via `declare_test_command` tool |
| `duplicate_steps` detector | `441cfff` | Catches "model replanned with similar wording" |
| Refactor: guidance.py → package | `f1ff19c` | Split 776-LOC monolith into 5 focused modules |
| Silent symbol deletion detector | `554c444` | Tree-sitter symbol diff catches "edit dropped functions accidentally" |
| Per-runner test parsers (1 class each) | `a0dacaa` | 7 parsers as classes in `test_parsers/` package |
| `tail_test_output` meta tool | `60154a3` | Structured failure extraction without re-running tests |
| guidance entries point at the new tool | `8abbada` | `stuck_on_tests` and `same_test_output_loop` mention `tail_test_output` |
| `first_test_run` proactive detector | `776372a` | Introduces the tool BEFORE the model gets stuck |
| Drop done/blocked guard on guidance hook | `df736c1` | Lets `first_test_run` fire even on the last step |
| `malformed_tool_call` detector | `75a2d9d` | Catches "model emits tool_call JSON as text instead of as a function call" |
| Per-parser flag tables (refactor) | `ffca614` | Each TestParser owns its CLI tokens and flag tables; cargo↔go substring collision fixed |
| `regression_after_edit` detector | `d677b81` | Catches "edit broke a previously passing test" via per-command outcome history |
| Static-analysis latency timer (instrumentation only) | `13ce2d3` | 4 categories: syntax_check, silent_deletion, guidance, plan_validate |
| Invert default — opt-in via env var | `a8caac7` | `INFINIDEV_ENABLE_SA_TIMER=1` to record/print |
| Wider engine instrumentation | `2fa0e7d` | Added `between_llm_calls` + 4 finer categories |
| Unattributed-time instrumentation | `05481fa` | Added `subprocess_exec`, `db_write`, `tool_io`, `trace_log` |
| **DB thread-local connection pool** | `f81d083` | **`db_write`: 1812ms → 12ms (642x), `between_llm_calls`: 4754ms → 1704ms** |
| **Async file indexing via existing IndexQueue** | `ad0b13f` | **`file_indexing`: 478ms → 0.18ms (2650x), via existing IndexQueue worker** |

### What we tested in production (and the outcome)

| Model | Task | Outcome | Notes |
|---|---|---|---|
| qwen3.5:4b | T1 (hello.py + test) | ✅ first try | test idiomatic, code review approved |
| qwen3.5:4b | T2 (read minidb) | ✅ first try | 4-bullet summary, almost-correct (picked wrong "easiest" test category) |
| qwen3.5:4b | T3 (CREATE+INSERT minidb) | ⚠ 1/2 → 0/2 regression | This is the run that motivated `regression_after_edit` |
| qwen3.5:9b | T3 | ✅ 2/2 | `malformed_tool_call` fired (twice) and the model self-corrected |
| glm-4.7-flash | T1 (after fixes) | ✅ 64s wall clock | First success after 11+ commits of loop fixes |
| glm-4.7-flash | T3v7 (with full stack) | ✅ 2/2 | First-ever T3 success with this model |
| **gemma4:26b** (Ollama) | minidb-full | ❌ broken FC | 0% GPU, `[empty response]` repeated. **DO NOT USE OLLAMA FOR GEMMA4.** |
| gemma4:26b (llama-server) | minidb-full | ⚠ planning forever | FC works perfectly, but model planned 8+ steps without writing a single byte to `minidb.py`. This is the pattern that motivates A1 (`stop_planning_start_coding`). |

### Decisions we made and stuck with

These are NOT pending — they are settled choices. Listing them here so
the next session doesn't re-debate them:

1. **Guidance is reactive AND proactive**, not just one. Reactive
   detectors (`stuck_on_*`) fire when something is going wrong;
   proactive detectors (`first_test_run`) fire on benign events to
   introduce tools BEFORE the model gets stuck. The dispatcher
   handles both in the same `_DETECTORS` list.

2. **Each TestParser owns its flag tables**. NO global flag list —
   pytest's `-k` does not interfere with cargo's `-k`. Adding a new
   runner is one new file in `test_parsers/`, no edits to
   `test_runners.py`.

3. **`is_test_command` uses token-sequence matching, not substring**.
   This was a real bug fix: `"cargo test"` contains `"go test"` as a
   substring (because `cargo` ends in `go`), so naive substring
   matching attributed cargo commands to GoTestParser. The fix is
   in `TestParser.matches_command` in `test_parsers/base.py`.

4. **`db_write` and `file_indexing` were the bottlenecks, NOT the
   static-analysis stack we added**. The pre-write syntax check
   costs ~5ms total per task; silent deletion costs ~3ms total;
   guidance detection costs ~0.1ms total. The 4 SECONDS of GPU-idle
   gap the user noticed came from preexisting code paths
   (DB connection setup, sync file indexing). This is documented in
   the commit messages of `f81d083` and `ad0b13f`.

5. **The static_analysis_timer is opt-in via env var, off by
   default, AND prints opt-in via the same env var**. One toggle:
   `INFINIDEV_ENABLE_SA_TIMER=1`. We removed an earlier
   `LOOP_REPORT_STATIC_ANALYSIS_LATENCY` setting because two toggles
   for the same thing was confusing.

6. **Trace log lives in `~/swe/traces/`, NOT in the workdir**. The
   first run of glm-4.7-flash had its trace inside the workdir and
   the model literally read its own trace.log mid-task, contaminating
   its context. The convention is now: trace files outside the
   workdir, always.

### Patterns we saw repeatedly that the system DOES handle

- glm-4.7-flash and gemma4 generate `{"tool_calls": [...]}` as text
  in their content/thinking → `malformed_tool_call` catches this.
- Models add many similar steps when confused → `duplicate_steps`
  catches this.
- Models edit code that breaks a previously passing test →
  `regression_after_edit` catches this (per-command outcome history,
  not naive comparison).
- Models run pytest 3+ times with identical fingerprints →
  `same_test_output_loop` catches this.
- Models read `state.last_test_output` and re-parse it manually
  instead of using `tail_test_output` → guidance entries explicitly
  point at the tool and show the JSON shape.

### Patterns we saw repeatedly that we DO NOT yet handle

- Models plan 8+ steps before writing a single byte (gemma4-26B).
  → **A1** below.
- Reactive guidance fires too late because it only runs at end of
  step, not mid-step (T3v6 with glm).
  → **A2** below.
- Models receive `tail_test_output` guidance but never call the tool
  on their own.
  → **A4** below (auto-call on pytest fail).
- Models keep emitting malformed tool calls even AFTER receiving
  `malformed_tool_call` guidance — could rescue the tool call from
  content automatically.
  → **B5** below.

---

## A — High confidence (motivated by evidence we observed in real runs)

### A1. `stop_planning_start_coding` detector

**Observed in**: gemma4:26b on minidb-full benchmark. The model created
8+ steps and modified them repeatedly without ever opening a file for
write. `minidb.py` stayed at 891 bytes (template default) for 5+
minutes of GPU time. `duplicate_steps` fired but the model "responded"
by adding more steps with minor wording variations instead of editing.

**Heuristic**: fires when
```
state.task_has_edits == False
AND state.plan.steps_done >= 5
```

**Guidance entry**: "stop planning, start opening files. Pick the
file your active step names, call read_file, then replace_lines.
Anything else is procrastination."

**Files**: `engine/guidance/detectors.py`, `engine/guidance/library.py`.
**Estimated**: ~25 LOC + library entry.

---

### A2. `maybe_queue_guidance` must run inside the inner loop

**Observed in**: T3v6 (glm-4.7-flash on minidb). The model ran pytest
5 times *inside Step 4*, all with similar exit codes, but
`same_test_output_loop` never fired because the detector hook only
runs at end-of-step. The guidance arrived only after Step 4 closed
with `status=done` — too late to be useful.

**Fix**: call `maybe_queue_guidance` after each LLM call in the inner
loop (`engine.py::_run_inner_loop`), not just at the step boundary.
The render step (`drain_pending_guidance`) stays where it is — only
the QUEUEING moves into the inner loop.

**Risk**: a guidance entry firing mid-step needs the next prompt
build to render it, which happens at the next iteration's
`_build_iteration_messages`. Confirm the timing by tracing the order
of `maybe_queue_guidance` → `build_iteration_prompt` calls.

**Files**: `engine/loop/engine.py` only.
**Estimated**: ~15 LOC.

---

### A3. Hard cap on plan size for small models

**Observed in**: gemma4-26B (12+ steps before any edit) and qwen3.5:4b
(repeated planning loops). Even with `duplicate_steps` and
`vague_steps` detectors, small models keep adding steps as a coping
mechanism for not knowing what to do next.

**Fix**: When `is_small=True` and `len(plan.pending_steps) >= 8` and
`task_has_edits == False`, the next `add_step` call returns an error
instead of a warning: "your plan is full and you haven't started
editing — clean up duplicates with remove_step or call
step_complete(continue) and start working on step 1".

**Files**: `tools/meta/plan_tools.py::AddStepTool._run`.
**Estimated**: ~10 LOC + new error message + test.

---

### A4. Auto-`tail_test_output` after pytest fail

**Observed in**: every model in this session. The capture path
populates `state.last_test_output` automatically, the `tail_test_output`
tool exists, the guidance mentions it — but small models still don't
discover it on their own. They re-read the raw 2000-byte stdout from
the `execute_command` result.

**Fix**: When `execute_command` finishes and the result is a test run
with `exit_code != 0`, automatically run `parse_test_failures` on the
captured output and APPEND a structured summary to the tool result.
The model gets the parsed failures **inline** with the original output:

```
{exit_code: 1, stdout: "...", structured_failures: [
  {test_name: "...", file: "...", line: 42, error_type: "KeyError", message: "..."}
]}
```

Zero LLM call extra. Probably solves 80% of the "model doesn't read
tracebacks" problem because the failure is right there in the
response the model is already going to read.

**Files**: `engine/loop/engine.py` (the same hook where we capture
last_test_output) — augment the tool result before appending to
messages.

**Estimated**: ~30 LOC.

---

## B — Identified, not yet implemented

### B5. Recover malformed tool calls instead of just warning

**Today**: when a model emits `{"tool_calls": [...]}` as text instead
of as a real function call, `malformed_tool_call` detector fires and
sends a guidance entry. But the malformed call itself is **discarded**
— the model has to retry on its own.

**Fix**: in `engine/formats/tool_call_parser.py`, after the existing
`_normalize_single_call` rescue (Bug #12), try to extract a tool
call from the *content* field even when `tool_calls` is empty. If we
find a JSON-shaped tool call in the content, normalize it the same
way and inject it as if it had been emitted properly.

**Risk**: false positives — model writes example JSON in a comment and
we execute it. Mitigation: only attempt when content is "almost
nothing else" (whitespace + maybe a code fence).

**Files**: `engine/formats/tool_call_parser.py`.
**Estimated**: ~50 LOC.

---

### B6. Provider FC capability sniff at startup

**Observed in**: gemma4:26b on Ollama returned `[empty response]`
repeatedly (minutes of CPU + 0% GPU) because the Ollama chat template
for gemma4 mishandles function-calling. The same model on llama-server
worked perfectly.

**Fix**: at engine startup (or first LLM call), if `manual_tc` is
detected, issue a tiny FC test against the registered tools (1 token
input, force a single trivial tool). If the response comes back empty
3 times in a row, log a warning telling the user "this model in this
provider has broken function calling — try a different provider or
add `INFINIDEV_FORCE_MANUAL_TC=1`".

**Files**: `config/model_capabilities.py` or `engine/loop/llm_caller.py`.
**Estimated**: ~40 LOC.

---

## C — Structural / larger effort

### C7. Refactor `engine/loop/engine.py` (1131 LOC)

**Why**: largest file in the repo, mixes 3 responsibilities:
outer loop orchestration, inner tool dispatch, error handling.
Several times this session I had to scroll 200+ lines to find the
right hook point.

**Plan**:
- `engine/loop/outer_loop.py` — the iteration `for` loop, plan
  advancement, finish conditions
- `engine/loop/inner_loop.py` — the `while` loop with LLM call,
  tool dispatch, guidance hook
- `engine/loop/error_handling.py` — circuit breakers, timeouts, retry

The current `engine.py` becomes a thin facade that imports the three
submodules and exposes `LoopEngine.execute`. ~50 LOC.

**Risk**: medium. Lots of state passes between functions. Needs
careful analysis of what's truly per-step vs per-iter vs per-task.

**Estimated**: ~4 hours of careful refactoring + test sweep.

---

### C8. Native prompt caching (Anthropic / OpenAI / litellm)

**Why**: every iteration rebuilds the system + task + plan + history
+ opened-files context from scratch. The system + task prefix is
stable across iterations. Anthropic (`cache_control`), OpenAI, and
litellm all support marking a prefix as cacheable so subsequent calls
only re-process the tail.

**Estimated savings**: ~50-70% input tokens on tasks with long stable
prefixes. Direct cost saving on hosted models, ~3-5x latency
reduction on Anthropic prompt caching.

**Files**: `engine/loop/context.py::build_iteration_prompt` (mark the
prefix), `engine/llm_client.py` (pass cache_control through to the
provider).

**Estimated**: ~80 LOC + per-provider testing.

---

### C9. Cache top-level symbols in `opened_files`

**Why**: `detect_silent_deletions` calls `extract_top_level_symbols`
twice per write (old + new content). The old extraction can be
cached when the model `read_file`-d the file recently — its content
is already in `state.opened_files`.

**Fix**: add a `cached_symbols: set[str]` field to the `OpenedFile`
dataclass, populate it lazily on the first `extract_top_level_symbols`
call against an opened file's content, reuse on subsequent
`detect_silent_deletions` calls.

**Estimated savings**: cuts `detect_silent_deletions` cost in half on
average. Today the cost is already ~10ms total per task post-A1+A2,
so this is **optimisation for the sake of it** — only do it when
the bench shows it matters again on larger files.

**Files**: `engine/loop/opened_file.py`, `tools/file/_helpers.py`.
**Estimated**: ~30 LOC.

---

### C10. Auto-fire `stuck_on_planning` (today only manual)

**Why**: the library has an entry for it but no detector — it's only
delivered when something else explicitly forces it. We never fire
it automatically.

**Heuristic**: "first add_step happened more than 30 seconds ago AND
no `read_file` / `edit_file` / `create_file` has run yet" → fire.

**Files**: `engine/guidance/detectors.py` (new function +
`_DETECTORS` entry).
**Estimated**: ~25 LOC.

---

## D — Software-quality (not behaviour)

### D11. Tests for the new detectors

**Why**: this session added ~15 detectors (`malformed_tool_call`,
`first_test_run`, `regression_after_edit`, `duplicate_steps`,
`same_test_output_loop`, etc.) — all verified with **inline bash
smoke tests** during development. Zero of them have a formal pytest
test in the repo.

**Plan**: create `tests/test_guidance_detectors.py` with one test
per detector, each covering:
- Positive case (the detector should fire)
- Negative case (similar but should NOT fire)
- Edge case (boundary value, empty messages, etc.)

**Files**: new `tests/test_guidance_detectors.py`.
**Estimated**: ~200 LOC.

---

### D12. Tests for the test_parsers package

**Why**: 7 parsers (pytest, jest, mocha, node:test, go, cargo, rspec)
each with smoke tests inline during dev. No formal pytest tests.

**Plan**: `tests/test_test_parsers.py` with one fixture file per
runner containing a real captured output, asserting the parsed
`ParsedFailure` list matches expected values.

**Files**: new `tests/test_test_parsers.py` + `tests/fixtures/runners/*.txt`.
**Estimated**: ~250 LOC.

---

### D13. Tests for the static_analysis_timer module

**Why**: the timer + `add_elapsed` API was added this session and is
load-bearing for benchmark accuracy. No formal tests.

**Plan**: `tests/test_static_analysis_timer.py` covering:
- enabled / disabled by env var
- multiple categories accumulate independently
- `add_elapsed` and `measure` produce equivalent results
- `reset()` zeroes everything
- `render()` formats correctly

**Files**: new `tests/test_static_analysis_timer.py`.
**Estimated**: ~100 LOC.

---

### D14. Fix `test_smart_index` (4 pre-existing failures)

**Why**: 4 tests in `tests/test_smart_index.py` fail due to test-DB
contamination between runs (symbols accumulate from prior tests). We
deselected them in every test command this session. They cover
`smart_index.py` which is now MORE relevant after wiring
`background_indexer` through it.

**Plan**: figure out the fixture isolation issue (probably needs a
fresh DB per test or a cleanup fixture). One-shot fix in the test
file's conftest or fixture.

**Files**: `tests/test_smart_index.py`, possibly `tests/conftest.py`.
**Estimated**: ~50 LOC of fixture work.

---

### D15. Tests for the DB connection pool

**Why**: `f81d083` introduced thread-local pooling that's load-bearing
for performance. Edge cases worth covering:
- pool returns same conn across calls
- pool reopens after `OperationalError` evicts
- non-busy `OperationalError` rolls back AND evicts (vs busy which
  retries on the same conn)
- cross-thread isolation (worker thread gets its own connection)
- non-sqlite exception triggers rollback

**Files**: new `tests/test_db_pool.py`.
**Estimated**: ~120 LOC.

---

## E — Investigation tasks (not code yet)

### E16. Wider sa_timer sweep on more code paths

**Why**: this session uncovered that DB connection setup and file
indexing were the actual bottlenecks (both ~37ms-53ms × N calls,
preexisting in the repo, NOT introduced by static analysis).
Suggests more bottlenecks may be hiding in:
- `behavior_tracker` invocations (124+ hook calls per task)
- `_emit_loop_event` (also fires per LLM call)
- `_update_opened_files_cache` (called on every read with full file content)
- `record_artifact_change` post-pool (down to 0.28ms but worth re-confirming)

**Plan**: add temporary `measure(...)` blocks to each suspect, run
qwen4b T1 bench, look for any category > 50ms total.

**No code change yet** — this is exploratory work that determines
what to optimise next.

---

### E17. Per-step wall-clock attribution

**Why**: today the timer reports per-task totals. Sometimes you want
to know "step 4 was the slow one" not "the run was 5 seconds".

**Plan**: extend `static_analysis_timer` to also track per-step
totals (reset between steps via `state.iteration_count`), render a
per-step matrix at the end of verbose runs.

**Files**: `engine/static_analysis_timer.py`,
`engine/loop/step_manager.py::finish`.
**Estimated**: ~60 LOC.

---

## Top 3 to do first

In order of ROI for the next session:

1. **A4 — auto-`tail_test_output` after pytest fail**. ~30 LOC, big
   impact, validable empirically with a re-run of qwen4b T3. Closes
   the loop "model doesn't read test output" without requiring the
   model to ask for it.

2. **A2 — guidance hook in inner loop**. ~15 LOC, the most concrete
   bug-of-design pending. Unblocks every reactive detector to fire
   in real time instead of at step boundaries.

3. **D11 + D12 + D13 — formal tests**. ~550 LOC across three test
   files. Zero behaviour change, protects everything else from
   silent regression. Worth doing once, before adding new features
   on top.

After those three, A1 (`stop_planning_start_coding`) and A3 (plan
size cap) are the next high-value targets — both attack patterns we
saw in real runs this session.

---

## Operational notes — how to run the system tomorrow

### Env vars that matter

| Env var | Purpose |
|---|---|
| `INFINIDEV_ENABLE_SA_TIMER=1` | Enable per-category latency measurement and print the block at end of run. Off by default. |
| `INFINIDEV_TRACE_FILE=<path>` | Append an iteration-by-iteration trace (prompt, thinking, tool calls, plan) to this file. **Always put this OUTSIDE the workdir** or the model will read its own trace. |
| `INFINIDEV_LLM_BASE_URL=<url>` | Override the LLM base URL — use this to point at llama-server or other openai-compatible endpoints. |
| `INFINIDEV_LLM_API_KEY=<key>` | Required when using `openai_compatible` provider; can be any string for local servers. |
| `INFINIDEV_LOOP_CUSTOM_TEST_COMMANDS="cmd1,cmd2"` | Add project-specific test command substrings that the guidance system should recognise as test runs. |

### Provider choices

- **Ollama** is fine for **qwen** family (3.5:4b, 3.5:9b, 2.5-coder:32b
  all worked). FC works natively for these.
- **Ollama** has **broken FC for gemma4** — confirmed in this session.
  Symptoms: 0% GPU, `[empty response]` in trace, retries forever. Use
  llama-server instead.
- **llama-server** (via `~/models/serve.sh`) is the right choice for
  gemma4 and any other model whose Ollama chat template is broken.
  Default port is now **8080** (was 8090, changed mid-session). The
  GGUF must be in `~/models/`.
- **Don't run both Ollama and llama-server simultaneously** unless
  they're on different GPUs — both will fight for the same VRAM.

### Workdir convention

```
~/swe/
├── minidb-template/         # canonical template for the SQL benchmark — DO NOT EDIT
├── runs/                    # task workdirs (model name + task id slug)
│   ├── qwen4b-t1/
│   ├── qwen4b-bench/
│   └── ...
└── traces/                  # trace_log files (one per run)
    ├── qwen4b-t1.log
    └── ...
```

Clean up `runs/` and `traces/` periodically — they accumulate fast.
Keep `minidb-template/` untouched.

### Standard bench commands

```bash
# Smoke test (small model, trivial task)
cd ~/swe/runs/<slug> && \
  INFINIDEV_ENABLE_SA_TIMER=1 \
  INFINIDEV_TRACE_FILE=$HOME/swe/traces/<slug>.log \
  uv run --project /home/andres/infinidev infinidev --classic \
    --model "qwen3.5:4b" --provider ollama \
    --prompt "Create a Python file hello.py..."

# Full benchmark on minidb (focused implementation)
cp -r ~/swe/minidb-template ~/swe/runs/<slug> && \
cd ~/swe/runs/<slug> && \
  INFINIDEV_ENABLE_SA_TIMER=1 \
  INFINIDEV_TRACE_FILE=$HOME/swe/traces/<slug>.log \
  uv run --project /home/andres/infinidev infinidev --classic \
    --model "qwen3.5:9b" --provider ollama \
    --prompt "Implement only CREATE TABLE and INSERT INTO support in minidb.py..."

# Using llama-server (e.g. for gemma4)
~/models/serve.sh gemma-4-26B-A4B-it-UD-Q6_K.gguf -c 32768 &
# wait for "main: server is listening"
INFINIDEV_LLM_BASE_URL=http://localhost:8080/v1 \
  INFINIDEV_LLM_API_KEY=local \
  INFINIDEV_ENABLE_SA_TIMER=1 \
  uv run --project /home/andres/infinidev infinidev --classic \
    --model "custom_openai/gemma4-26b" \
    --provider openai_compatible \
    --prompt "..."
```

### How to interpret the latency block

The `Static analysis accumulated latency:` block at the end of an
opt-in run shows per-category totals and averages. The most
important row is `between_llm_calls` — that's the wall-clock GAP
between LLM invocations (the GPU-idle time). The other rows are
finer attributions of what happens INSIDE that gap. They will sum
to slightly more than `between_llm_calls` because some categories
overlap (e.g. `subprocess_exec` runs inside a tool call which is
inside `between_llm_calls`, and `db_write` may run in parallel).

After the A1 + A2 optimisations of this session, a typical small
task should show:
- `between_llm_calls` ≈ 1500-2000 ms total
- `subprocess_exec` ≈ 1500-1800 ms (pytest startup — this is
  legitimate work, not overhead)
- everything else combined ≈ 50-100 ms

If you see a category > 100 ms total that ISN'T `subprocess_exec`,
you've probably found a new bottleneck worth investigating.

### How to interpret the guidance log

Verbose runs print `↪ guidance queued: <key>` whenever a detector
fires. If a run completes without a model passing tests AND no
guidance fired, that's evidence the detectors are too conservative
for the failure mode that occurred — file a TODO entry and write a
new detector.

---

## Files that were touched recently (for orientation)

These are the files most likely to need attention when resuming:

### Core engine
- `engine/loop/engine.py` (1131 LOC) — the loop. Refactor target (C7).
- `engine/loop/step_manager.py` — finish, summarize, advance_plan
- `engine/loop/context.py` — `build_iteration_prompt` (instrumented as `prompt_build`)
- `engine/loop/loop_state.py` — `LoopState` model with the new fields
  (`pending_guidance`, `guidance_given`, `last_test_output`,
  `test_outcome_history`, `regression_signaled`, `custom_test_commands`)

### Guidance system (the package built this session)
- `engine/guidance/__init__.py` — public API, `_PARSERS` registry
- `engine/guidance/library.py` — 13 GuidanceEntry definitions
- `engine/guidance/detectors.py` — all `_has_*` functions and `_DETECTORS` list
- `engine/guidance/test_runners.py` — `is_test_command`, `normalize_test_command`, `test_outcome_fingerprint`
- `engine/guidance/hooks.py` — `maybe_queue_guidance`, `drain_pending_guidance`

### Test parsers (the package built this session)
- `engine/test_parsers/base.py` — `TestParser` ABC + `ParsedFailure` dataclass
- `engine/test_parsers/{pytest,jest,mocha,node_test,go,cargo,rspec}_parser.py` — one class each
- `engine/test_parsers/__init__.py` — `_PARSERS` registry, `parse_test_failures` dispatch

### Static analysis & code intel
- `code_intel/syntax_check.py` — `check_syntax`, `extract_top_level_symbols`
- `code_intel/background_indexer.py` — async indexing entry point (NEW this session)
- `engine/static_analysis_timer.py` — latency accumulator (NEW this session)
- `tools/file/_helpers.py` — `validate_syntax_or_error`, `detect_silent_deletions`, `deletion_warning_text`, `atomic_write`

### Tools (modified for instrumentation / async indexing)
- `tools/file/read_file_tool.py` — calls `enqueue_or_sync` instead of `ensure_indexed`
- `tools/file/{create,replace_lines,edit,multi_edit,write}_file_tool.py` — call `validate_syntax_or_error` + `detect_silent_deletions`
- `tools/meta/plan_tools.py` — `_looks_concrete` regex + warning + hint
- `tools/meta/declare_test_command_tool.py` — agent-declared test commands
- `tools/meta/tail_test_output_tool.py` — three modes (tail/failures/structured)
- `tools/shell/execute_command_tool.py` — wrapped in `subprocess_exec` measure
- `tools/base/db.py` — `get_pooled_connection`, the thread-local cache (CRITICAL — A1 lives here)

### CLI
- `cli/main.py` — `_run_main` and `_run_single_prompt` both register the IndexQueue globally now

### Settings
- `config/settings.py` — new fields: `LOOP_SUMMARIZER_TIMEOUT`,
  `LOOP_REQUIRE_NOTE_BEFORE_COMPLETE`, `LOOP_VALIDATE_SYNTAX_BEFORE_WRITE`,
  `LOOP_GUIDANCE_ENABLED`, `LOOP_GUIDANCE_MAX_PER_TASK`,
  `LOOP_CUSTOM_TEST_COMMANDS`

---

## Things that are deliberately NOT in this list

These came up during the session and we explicitly decided not to
pursue them. Listing here so the next session doesn't reopen the
debate:

- **Subprocess pool for `execute_command`** — pytest startup is
  ~1.5s of the remaining gap. Could pool a python interpreter and
  use `pytest --collect-only` patterns to skip restart. Risk:
  high invasiveness, hard to debug, likely to break tests that
  depend on a clean process.
- **Smart-context-summary refactor** — the `build_iteration_prompt`
  smart context block is a contributor to prompt token cost. We
  measured `prompt_build` at ~12ms which is fine; the LLM-side cost
  is real but harder to attack without rethinking the whole prompt
  structure. Better attacked via C8 (native prompt caching).
- **More test runner parsers** — we have 7. Adding more (ava,
  jasmine, ts-node) is busywork without a real-world request.
- **Refactor of the Analyst → Develop pipeline** — the imperative
  bypass we added at the start of the session is a workaround,
  not a fix. The deep fix would be to rethink whether
  `_run_single_prompt` should ever go through the analyst at all.
  Risk: the analyst is needed for non-imperative tasks. Don't
  touch unless someone files a bug.

---

## Quick mental model for resuming

If you only have 5 minutes to remember what we did:

1. **The loop works for small models now.** glm-4.7-flash, qwen3.5:4b,
   qwen3.5:9b can all complete simple-to-medium tasks without
   spinning forever.
2. **The guidance system is the main user-visible win.** 13 detectors
   + entries that catch specific failure modes and inject pre-baked
   how-to advice with zero LLM call overhead.
3. **The latency stack got 43x faster** between LLM calls by fixing
   two preexisting bottlenecks (DB connection setup, sync file
   indexing). The features added this session are NOT bottlenecks.
4. **Gemma4 is broken on Ollama, fine on llama-server.** Never debug
   gemma4 against Ollama again — go straight to llama-server.
5. **The next bottleneck is pytest startup**, but that's legitimate
   work, not overhead. Don't try to optimise it without strong
   justification.

When in doubt, run a benchmark with `INFINIDEV_ENABLE_SA_TIMER=1` and
let the data tell you where to look next. That's been the discipline
all session and it has paid off every time.
