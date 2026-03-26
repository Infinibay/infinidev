# Phase-Based Execution System Design

## Problem

The current system lets the model plan and execute freely. With small local
models (7B-35B), this results in:
- Vague 2-3 step plans ("Implement everything")
- Attempts to write 800+ lines in a single tool call
- Regressions (17/61 tests → 7/61 after rewriting)
- No verification between edits
- Constant re-reading of files already read

## Core Insight

Small models are **bad at planning** but **good at following specific
instructions**. The engine controls flow, the model fills in the content.

---

## Architecture

```
                    ┌─────────────┐
                    │  Classifier  │  (already exists: TicketType)
                    │  bug/feature │
                    │  refactor/.. │
                    └──────┬──────┘
                           │
                           ▼
                ┌──────────────────┐
                │  Strategy Router  │  picks phase config by task type
                └────────┬─────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────────┐
    │ ANALYZE  │──→│   PLAN   │──→│   EXECUTE    │
    │ (read)   │   │ (think)  │   │ (build+test) │
    └──────────┘   └──────────┘   └──────────────┘
```

---

## Strategy Configs by Task Type

Each task type defines:
- Whether to run ANALYZE phase (and how deep)
- How to prompt each phase
- Step size constraints
- Verification strategy

### Bug Fix Strategy

```yaml
analyze:
  goal: "Reproduce the bug and locate the root cause"
  instructions:
    - "Run the failing test(s) or reproduce the error"
    - "Read the error output carefully"
    - "Locate the function(s) where the error originates"
    - "Trace the logic to find the root cause"
    - "Note: file, function, line, what's wrong, what it should do"
  tools: read-only + execute_command
  max_tool_calls: 15

plan:
  constraints:
    - "Each step fixes ONE specific issue in ONE function"
    - "Include a verification step after each fix (run the specific failing test)"
    - "Do NOT refactor or improve code unrelated to the bug"
  min_steps: 2
  max_step_files: 2

execute:
  verification: "run failing test after each code change"
  scope_rule: "only touch the file(s) and function(s) in the step"
```

### Feature Implementation Strategy

```yaml
analyze:
  goal: "Understand the existing codebase patterns and integration points"
  instructions:
    - "Identify existing patterns (naming, structure, error handling)"
    - "Find where the new feature integrates with existing code"
    - "Check if similar features exist that can be used as reference"
    - "Note: patterns to follow, files to modify, interfaces to implement"
  tools: read-only + execute_command
  max_tool_calls: 20

plan:
  constraints:
    - "Start with the smallest working skeleton (stub/minimal implementation)"
    - "Build incrementally — each step adds one capability"
    - "Each step should result in more tests passing or a working sub-feature"
    - "Include 'run tests' step after every 2-3 implementation steps"
    - "Order: foundation → core logic → edge cases → polish"
  min_steps: 5
  max_step_files: 2

execute:
  verification: "run test suite after each code-modifying step"
  scope_rule: "only touch the file(s) and function(s) in the step"
  anti_rewrite: true  # warn if model uses write_file on existing file
```

### Refactor Strategy

```yaml
analyze:
  goal: "Map the code to refactor and its dependencies"
  instructions:
    - "Identify all callers/consumers of the code being refactored"
    - "Run full test suite to establish baseline"
    - "Note: all files that import/use the target code"
    - "Note: current test coverage and passing count"
  tools: read-only + execute_command
  max_tool_calls: 20

plan:
  constraints:
    - "Each step is a single, atomic refactoring move"
    - "Tests MUST pass after every step (behavior-preserving)"
    - "Never change behavior and structure in the same step"
    - "Include 'run full test suite' after every step"
  min_steps: 3
  max_step_files: 3

execute:
  verification: "run FULL test suite after EVERY step (not just related tests)"
  scope_rule: "only the refactoring described in the step"
  regression_threshold: 0  # ANY test regression = immediate warning
```

### Implement From Scratch Strategy

```yaml
analyze:
  goal: "Understand the full specification before writing any code"
  instructions:
    - "Read ALL specification files (tests, docs, requirements)"
    - "For each test class/group, note what it expects"
    - "Identify the public API (classes, methods, signatures)"
    - "Identify dependencies between components"
    - "Note: component list, dependency order, API surface"
  tools: read-only + execute_command
  max_tool_calls: 20

plan:
  constraints:
    - "First step: create the file with class skeleton and stub methods"
    - "Order implementation by dependency (foundations first)"
    - "Each step implements ONE method or ONE small group of related methods"
    - "Each step should make 2-5 more tests pass"
    - "Include 'run tests and note progress' after every implementation step"
    - "Do NOT rewrite the file — always use edit_file to ADD to it"
  min_steps: 8
  max_step_files: 2

execute:
  verification: "run test suite after every step, report pass count"
  scope_rule: "only implement what the step says, nothing more"
  anti_rewrite: true
  track_progress: true  # show [Tests: 15/61 ↑5] after each step
```

### Sysadmin / Other Strategy

```yaml
analyze:
  goal: "Understand the current system state"
  instructions:
    - "Check relevant system state (files, configs, services)"
    - "Note: current state, expected state, what needs to change"
  tools: read-only + execute_command
  max_tool_calls: 10

plan:
  constraints:
    - "Each step is one specific action"
    - "Include verification after each change"
  min_steps: 1
  max_step_files: 3

execute:
  verification: "verify the change took effect"
  scope_rule: "only the action in the step"
```

---

## Phase Details

### Phase 1: ANALYZE

**Generic prompt structure** (strategy fills in the specifics):

```
You are in ANALYSIS MODE. Read and understand before acting.

Task: {task_description}

Your goal for this phase: {strategy.analyze.goal}

Instructions:
{strategy.analyze.instructions — joined as bullet list}

RULES:
- You can ONLY use read-only tools. No file modifications.
- You MUST use add_note after each significant finding.
- When you have enough understanding, call phase_complete.
- Notes should be structured: "FILE: path — FINDING: what you found"

Tool budget: {strategy.analyze.max_tool_calls} tool calls max.
```

The engine enforces the read-only restriction by filtering the tool set
before passing it to the loop.

### Phase 2: PLAN

**Generic prompt structure:**

```
Based on your analysis, create an implementation plan.

Task: {task_description}

Your findings:
{state.notes — numbered list}

{strategy-specific baseline info, e.g. "Test baseline: 0/61 passing"}

Plan requirements:
{strategy.plan.constraints — joined as bullet list}

Each step must specify:
- What to do (specific action)
- Which file(s) to modify
- Expected outcome

BAD: "Implement the feature" / "Fix the bugs" / "Write the code"
GOOD: "Add _parse_where() method to Database class in minidb.py —
       handle =, !=, >, <, >=, <= operators"

Output a JSON array of steps:
[
  {"step": 1, "description": "...", "files": ["..."]},
  ...
]
```

**Engine-side validation:**
- `len(steps) >= strategy.plan.min_steps` → reject if too few
- Each step has `description` (min 20 chars) and `files` (non-empty)
- Each step's `files` list has ≤ `strategy.plan.max_step_files` entries
- Has verification steps at required intervals
- If invalid → re-prompt with: "Your plan was rejected because: {reasons}. Fix it."
- Max 2 re-prompt attempts before falling back to simple loop

### Phase 3: EXECUTE

**Per-step prompt structure:**

```
STEP {n}/{total}: {step.description}

Files you may modify: {step.files}

Your analysis notes:
{state.notes}

Completed steps:
{summaries of done steps}

{progress info if applicable, e.g. "Tests: 15/61 passing (↑5 from last step)"}

RULES:
- ONLY modify the files listed above for this step
- Use edit_file to make changes (NOT write_file on existing files)
- Make the smallest change that achieves this step's goal
- Call step_complete when done with a summary of what you changed
```

**Engine-side enforcement after each step:**
1. Check if modified files match step's file list → warn if not
2. If code was modified → auto-run verification command
3. Track test progress → inject into next step's prompt
4. If regression detected → inject warning

---

## Test Checkpoint System

```python
@dataclass
class TestCheckpoint:
    command: str        # "python -m pytest test_minidb.py --tb=no -q"
    baseline: int = 0   # tests passing at start
    current: int = 0    # tests passing now
    high_water: int = 0  # max tests ever passed
    total: int = 0       # total test count

    def run(self) -> tuple[int, int]:
        """Run tests, parse output, return (passed, total)."""
        ...

    def progress_str(self) -> str:
        """e.g. 'Tests: 23/61 (↑5 from last step)'"""
        ...
```

**Auto-detection of test command:**
- Look for pytest.ini / pyproject.toml → `pytest`
- Look for package.json → `npm test`
- Look for Makefile with `test` target → `make test`
- Fallback: try `pytest` then `python -m pytest`

---

## Skip Logic: When NOT to Use Phases

Not every task needs 3 phases:

1. **Simple/conversational**: "What does this function do?" → direct loop, no phases
2. **Tiny fix**: "Change the timeout from 30 to 60" → direct loop
3. **Already specified**: User gave exact steps → skip ANALYZE, light PLAN

The classifier already returns `action: "passthrough"` for simple tasks.
For `action: "proceed"`, the engine checks task complexity:

```python
def needs_phases(analysis: AnalysisResult) -> bool:
    if analysis.action == "passthrough":
        return False
    # /think command forces phases
    if force_think:
        return True
    # Heuristic: complex if specification mentions multiple files/components
    spec = analysis.specification
    if len(spec.get("files_involved", [])) >= 3:
        return True
    if spec.get("complexity", "low") in ("medium", "high"):
        return True
    return False
```

---

## Implementation Plan

1. **`engine/phase_prompts.py`** — Strategy configs + prompt templates per task type
2. **`engine/test_checkpoint.py`** — Test runner, progress tracking, regression detection
3. **`engine/plan_validator.py`** — Validate model-generated plans against strategy constraints
4. **`engine/phase_engine.py`** — Orchestrator: classify → route → ANALYZE → PLAN → EXECUTE
5. **Integration** — Wire into CLI, `/think` activates full phases
6. **Benchmarks** — Re-run all challenges, compare results
