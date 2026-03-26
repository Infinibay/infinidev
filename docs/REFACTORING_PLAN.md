# Loop Engine Refactoring Plan

## Goal

Decompose `loop_engine.py` (2417 lines, ~40 functions in one file) into focused modules
so that adding a **phase-based execution system** (ANALYZE → PLAN → EXECUTE) is clean
and maintainable.

## Current State

```
loop_engine.py (2417 lines) — contains EVERYTHING:
├── JSON utilities (_safe_json_loads)
├── LLM calling + retry logic (_call_llm, _is_transient, _is_malformed)
├── Tool call parsing — 9 formats (~300 lines)
├── Tool execution + parallel batching (~200 lines)
├── File change tracking hooks (~100 lines)
├── Opened files cache management (~80 lines)
├── Logging — 7 functions (~70 lines)
├── Event emission helpers
├── Step summarization (_summarize_step)
├── Plan management (inline in execute())
├── LoopEngine.execute() — THE loop (~700 lines)
└── Guardrail validation loop
```

## Target State

```
engine/
├── base.py                    # AgentEngine ABC (exists, keep as-is)
├── models.py                  # LoopState, LoopPlan, StepResult, etc. (exists as loop_models.py)
├── context.py                 # Prompt building (exists as loop_context.py)
├── tools.py                   # Tool schemas + dispatch (exists as loop_tools.py)
├── summarizer.py              # Step summarization (exists, keep as-is)
├── file_change_tracker.py     # File diff tracking (exists, keep as-is)
├── flows.py                   # Flow configs (exists, keep as-is)
│
│   NEW MODULES (extracted from loop_engine.py):
│
├── llm_client.py              # LLM calling, retry, mode detection, /no_think
├── tool_call_parser.py        # All 9 TC parsers + _ManualToolCall + _safe_json_loads
├── tool_executor.py           # Batching, parallel exec, file hooks, cache updates
├── logging.py                 # All _log_* functions + ANSI codes + event emission
├── loop_engine.py             # ONLY the loop: init → iterate → finalize (~400 lines)
│
│   FUTURE (after refactoring, for phase system):
│
├── phases/
│   ├── __init__.py
│   ├── base.py                # Phase ABC with tool restrictions
│   ├── analyze.py             # Read-only phase: explore codebase, take notes
│   ├── plan.py                # Planning phase: generate granular plan from notes
│   └── execute.py             # Execution phase: scoped per-step implementation
└── phase_engine.py            # Orchestrates phases (replaces or wraps loop_engine)
```

## Refactoring Steps

### Phase 1: Extract `llm_client.py` (LOW RISK)

**What moves:**
- `_call_llm()` (lines 1016-1075) — core LLM calling with retry
- `_is_transient()` (lines 641-653) — transient error detection
- `_is_malformed_tool_call()` (lines 671-674) — malformed TC detection
- `_TRANSIENT_ERRORS`, `_PERMANENT_ERRORS`, `_MALFORMED_TOOL_PATTERNS` — constants
- `_LLM_RETRIES`, `_LLM_RETRY_DELAY` — retry config
- `/no_think` injection logic (currently inline in `_call_llm`)

**New interface:**
```python
class LLMClient:
    def call(self, params, messages, tools=None, tool_choice="auto") -> LLMResponse
    def is_transient(self, exc) -> bool
    def is_malformed(self, exc) -> bool
```

**Dependencies:** `litellm`, `model_capabilities` (lazy import)
**Dependents:** `loop_engine.execute()`, `_summarize_step()`

---

### Phase 2: Extract `tool_call_parser.py` (LOW RISK)

**What moves:**
- `_ManualToolCall` class (lines 677-695)
- `_parse_text_tool_calls()` (lines 698-840) — main parser with 9 formats
- `_parse_search_replace_blocks()` (lines 843-881)
- `_extract_calls_from_fragments()` (lines 884-925)
- `_extract_calls_from_array()` (lines 928-936)
- `_normalize_call_list()` (lines 939-947)
- `_normalize_single_call()` (lines 950-975)
- `_parse_step_complete_args()` (lines 978-1013)
- `_safe_json_loads()` (lines 48-65) — used by parsers + other places

**New interface:**
```python
def parse_tool_calls(content: str) -> list[ToolCall] | None
def parse_step_complete(arguments: str | dict) -> StepResult
def safe_json_loads(text: str) -> Any
class ManualToolCall:  # same as _ManualToolCall
```

**Dependencies:** `json`, `re`, `json_repair` (optional)
**Dependents:** `loop_engine.execute()` (FC fallback + manual mode)

---

### Phase 3: Extract `tool_executor.py` (MEDIUM RISK)

**What moves:**
- `_batch_tool_calls()` (lines 374-402) — batching strategy
- `_execute_tool_calls_parallel()` (lines 405-444) — ThreadPoolExecutor
- `_extract_file_path_from_args()` (lines 448-456)
- `_capture_pre_content()` (lines 459-481)
- `_extract_reason_from_args()` (lines 484-492)
- `_maybe_emit_file_change()` (lines 495-546)
- `_update_opened_files_cache()` (lines 273-354) — file cache management
- `_reindex_if_enabled()` (lines 262-270)

**New interface:**
```python
class ToolExecutor:
    def __init__(self, file_tracker, state, event_emitter)
    def execute_batch(self, tool_calls, tool_dispatch) -> list[ToolResult]
    def update_cache(self, tool_name, arguments, result, state)
```

**Dependencies:** `concurrent.futures`, `loop_tools.execute_tool_call`, `FileChangeTracker`
**Dependents:** `loop_engine.execute()` inner loop

**Why MEDIUM risk:** This touches the parallel execution pipeline and file
change tracking hooks. Needs careful testing to preserve batching semantics
and file diff accuracy.

---

### Phase 4: Extract `engine_logging.py` (LOW RISK)

**What moves:**
- All `_log_*` functions (7 total, lines 549-611)
- `_emit_loop_event()` (lines 110-117)
- `_emit_log()` (lines 147-159)
- `_log()` (lines 139-144)
- ANSI color constants (`_DIM`, `_BOLD`, etc.)
- `_STATUS_ICON` dict
- `_TOOL_DETAIL_KEYS` dict
- `_extract_tool_detail()` (lines 200-229)
- `_extract_tool_error()` (lines 232-257)

**New interface:**
```python
class EngineLogger:
    def log(self, msg)
    def log_start(self, agent_id, agent_name, role, desc, tool_count)
    def log_step_start(self, iteration, step_desc)
    def log_tool(self, agent_name, iteration, tool_name, call_num, total)
    def log_step_done(self, iteration, status, summary, tool_calls, tokens)
    def log_plan(self, plan)
    def log_finish(self, agent_name, status, iterations, total_tools, total_tokens)
    def emit_event(self, event_type, project_id, agent_id, data)
    def extract_tool_detail(self, tool_name, arguments) -> str
    def extract_tool_error(self, result) -> str
```

**Dependencies:** `event_bus` from flows/event_listeners
**Dependents:** Everything in loop_engine

---

### Phase 5: Slim down `loop_engine.py` (MEDIUM RISK)

After phases 1-4, `loop_engine.py` should only contain:
- `LoopEngine` class
- `execute()` method using the extracted modules
- `_apply_guardrail()` method
- `_synthesize_final()` helper
- `_get_model_max_context()` helper

**Target: ~400-500 lines** (down from 2417)

The execute() method becomes:
```python
def execute(self, agent, task_prompt, ...):
    # Init
    state = self._init_state(resume_state)
    llm = LLMClient(llm_params)
    parser = ToolCallParser()
    executor = ToolExecutor(file_tracker, state, logger)

    for iteration in range(max_iterations):
        # Build prompt
        prompt = build_iteration_prompt(state, ...)
        messages = [system_msg, user_msg]

        # Inner loop: tool calling within one step
        while action_tool_calls < max_per_action:
            response = llm.call(messages, tools)
            tool_calls = parser.extract(response, manual_mode)
            if not tool_calls:
                # handle text-only response
                ...
            results = executor.execute_batch(tool_calls, tool_dispatch)
            step_result = self._process_step_complete(results)
            if step_result:
                break

        # Post-step: summarize, update plan, check termination
        ...
```

---

### Phase 6: Verify everything works (CRITICAL)

- Run full test suite: `uv run pytest tests/ --ignore=tests/test_tui.py`
- Run benchmark tests (challenge2 with qwen3.5:35b)
- Verify TUI mode still works
- Verify classic CLI mode still works
- Check all event emissions still reach TUI

---

## Execution Order & Risk Assessment

| Phase | Risk | Lines Moved | Estimated Effort | Can Break Things? |
|-------|------|-------------|-----------------|-------------------|
| 1. llm_client | LOW | ~120 | Small | Only if retry logic changes |
| 2. tool_call_parser | LOW | ~350 | Medium | Only if parser interfaces change |
| 4. engine_logging | LOW | ~150 | Small | Unlikely (pure output) |
| 3. tool_executor | MEDIUM | ~250 | Medium | Parallel exec + file tracking |
| 5. slim loop_engine | MEDIUM | Rewrite | Large | Integration of all modules |
| 6. verify | CRITICAL | 0 | Medium | N/A — just testing |

**Recommended order: 1 → 2 → 4 → 3 → 5 → 6**

Start with the lowest-risk extractions to build confidence, then tackle the
more coupled tool_executor, and finally rewrite the slim loop_engine.

## Rules

- **No behavior changes.** This is a pure structural refactoring.
- **Tests must pass after each phase.** Don't batch phases.
- **Imports stay lazy** where they already are (litellm, httpx, etc.)
- **Public interface of LoopEngine doesn't change.** Same execute() signature,
  same return values, same events emitted.
- **Each phase gets its own commit.**

## After Refactoring: Phase System

With the clean module structure, adding phases becomes straightforward:

```python
# phase_engine.py
class PhaseEngine:
    def execute(self, agent, task_prompt, ...):
        state = LoopState()

        # Phase 1: ANALYZE — read-only tools, take notes
        analyze = AnalyzePhase(llm, parser, executor)
        analyze.run(state, allowed_tools=READ_ONLY_TOOLS)

        # Phase 2: PLAN — no tools, generate granular plan from notes
        planner = PlanPhase(llm)
        planner.run(state)  # produces state.plan with 5-15 small steps

        # Phase 3: EXECUTE — all tools, scoped per step
        execute = ExecutePhase(llm, parser, executor)
        execute.run(state, allowed_tools=ALL_TOOLS)

        return state.final_answer
```

Each phase controls which tools are available, what the prompt says,
and how many iterations/tool calls are allowed.
