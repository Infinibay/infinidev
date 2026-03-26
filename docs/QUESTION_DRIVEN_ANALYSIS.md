# Question-Driven Analysis System

## Summary

Replace the free-form ANALYZE phase with a structured question-answer flow:
1. Model generates questions about the task
2. Engine creates 1 step per question (read-only tools)
3. Answers accumulate as structured notes
4. Notes feed into PLAN phase

This makes `/think` unnecessary — every task goes through this flow.
Simple tasks get 1-2 quick questions. Complex tasks get 8-10.

## Flow

```
Task arrives
     │
     ▼
┌──────────────┐
│  QUESTIONS   │  Direct LLM call, no tools
│  "What do I  │  Model generates 3-10 questions
│  need to     │  Engine validates (not vague, not too many)
│  know?"      │  Examples of good/bad questions in prompt
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ INVESTIGATE  │  1 LoopEngine run per question
│  Q1: ...     │  Read-only tools only
│  Q2: ...     │  Max 6 tool calls per question
│  Q3: ...     │  Answer → add_note
│  ...         │  All notes carry forward
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    PLAN      │  Direct LLM call, no tools
│  Generate    │  Input: task + all answers
│  steps from  │  Output: validated JSON plan
│  answers     │  Engine rejects vague/large steps
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   EXECUTE    │  LoopEngine per step
│  Step 1      │  All tools available
│  Step 2      │  Scoped to step's files
│  ...         │  Test checkpoint after code changes
│  Step N      │  Regression detection
└──────────────┘
```

## Phase 0: QUESTION GENERATION

### What happens
- Direct `call_llm()` — no tools, no loop
- Model receives: task description + task type + examples
- Model outputs: JSON array of questions
- Engine validates: count, specificity, relevance

### Prompt structure
```
You are preparing to work on a task. Before starting, you need to
understand the problem fully. Generate questions that, once answered,
will give you everything you need to create a detailed plan.

Task: {description}
Task type: {task_type}

{examples of good and bad questions for this task type}

Output a JSON array of questions:
[
  {"question": "...", "intent": "understand_spec|find_code|check_tests|..."},
  ...
]
```

### Examples by task type

#### Bug fix
```
GOOD questions:
- "What error message or failing test reproduces this bug?"
- "What function contains the buggy code, and what file is it in?"
- "Are there existing tests covering this behavior?"
- "What is the expected behavior vs actual behavior?"

BAD questions:
- "What is the project about?" (too broad, not relevant to the bug)
- "How does the entire codebase work?" (not targeted)
- "What programming language is this?" (obvious from files)
```

#### Feature / implement from scratch
```
GOOD questions:
- "What is the full public API expected? (classes, methods, signatures)"
- "What test file defines the expected behavior? How many tests?"
- "Are there existing patterns in the codebase to follow? (imports, naming, structure)"
- "What are the dependencies between components? (what must be built first)"
- "What is the current test baseline? (how many pass now)"

BAD questions:
- "What should I implement?" (the task already says this)
- "Is there documentation?" (go look instead of asking)
```

#### Refactor
```
GOOD questions:
- "What is the current test baseline? (must not regress)"
- "What files/functions need to change?"
- "Who calls the code being refactored? (all dependents)"
- "Are there integration points that could break?"

BAD questions:
- "What does the code do?" (too vague — ask about specific functions)
```

### Validation rules
- Minimum 2 questions (anything less means the model didn't think)
- Maximum 10 questions (more wastes tool budget)
- Each question must be > 15 characters
- Reject questions that repeat the task description verbatim
- Reject overly generic questions: "What is the project?", "How does it work?"
- If validation fails: re-prompt once with feedback
- If still fails: fall back to 3 default questions per task type

### Default fallback questions (if model can't generate good ones)

Bug:
1. "Run the failing tests and note the exact error messages"
2. "Read the files mentioned in the error and find the buggy code"
3. "Check if there are related tests that still pass"

Feature:
1. "Read the test/spec files to understand what's expected"
2. "Check if similar code already exists in the project as reference"
3. "Run existing tests to establish a baseline"

Refactor:
1. "Run the full test suite and note the pass count"
2. "Read the code to refactor and identify all its callers"
3. "Check if the code has existing tests"

## Phase 1: INVESTIGATE

### What happens
- For each question from Phase 0, run a mini LoopEngine
- Read-only tools only
- Max 6 tool calls per question, max 2 iterations
- The model investigates and MUST call `add_note` with the answer
- Notes accumulate across all questions (shared state)

### Per-question prompt
```
QUESTION {n}/{total}: {question.text}

Investigate this question using the available tools. When you have
the answer, save it with add_note and call step_complete.

Previous answers:
{notes from answered questions so far}

RULES:
- Read files, search code, run commands — but do NOT modify anything
- You MUST call add_note with your answer before calling step_complete
- Be specific: include file names, line numbers, function names
- Keep your note concise but complete (2-4 sentences max)
```

### Key details
- State (notes) persists across questions — later questions see earlier answers
- If a question has no useful answer (e.g. "no tests found"), note that too
- Tool budget is per-question, not global — prevents one question eating all budget
- After all questions answered, engine collects all notes for PLAN phase

## Phase 2: PLAN (unchanged from current design)

### Input enrichment
The plan prompt now receives much richer context:
```
## YOUR INVESTIGATION RESULTS
  1. Q: "What is the full API expected?" A: "Database class with execute(),
     get_tables(), get_schema(). 61 tests in test_minidb.py..."
  2. Q: "What is the current baseline?" A: "0/61 tests passing. File does
     not exist yet."
  3. Q: "What dependencies exist?" A: "Table storage needed before INSERT,
     INSERT before SELECT, WHERE before JOINs..."
  ...
```

This gives the model concrete facts to build a plan from, instead of
vague memories from a free-form exploration.

### Same validation as before
- Min steps enforced
- No vague descriptions
- Test steps required
- Re-prompt on failure

## Phase 3: EXECUTE (unchanged from current design)

Same per-step execution with test checkpoints and regression detection.

## /think command — still useful?

With this system, every task gets question-driven analysis. But we could
still use /think to control the depth:

- **Normal mode**: Skip question generation entirely. Use old LoopEngine.
  Good for: "fix this typo", "what does this function do?"

- **`/think` mode**: Full question-driven flow (QUESTIONS → INVESTIGATE → PLAN → EXECUTE).
  Good for: complex features, implement from scratch, multi-file refactors

The trigger can be automatic based on the analysis_engine classification:
- `action: "passthrough"` → normal mode (no phases)
- `action: "proceed"` + simple task → normal mode
- `action: "proceed"` + complex task or `/think` → phase mode

## Implementation plan

### Step 1: Question generation prompt (`phase_prompts.py`)
- Add question generation prompts per task type
- Include 3-4 good/bad examples each
- Define fallback questions per type

### Step 2: Question validator (`plan_validator.py`)
- Add `validate_questions()` function
- Check: count, length, not too vague, no repeats
- Return (is_valid, questions, errors)

### Step 3: Update PhaseEngine (`phase_engine.py`)
- Replace `_run_analyze()` with `_generate_questions()` + `_investigate()`
- `_generate_questions()`: direct LLM call → validated question list
- `_investigate()`: for each question, run mini LoopEngine with read-only tools
- Notes from investigate feed into `_run_plan()` as Q&A pairs

### Step 4: Remove old ANALYZE prompts (`phase_prompts.py`)
- Remove `_BUG_ANALYZE`, `_FEATURE_ANALYZE`, `_REFACTOR_ANALYZE`, `_OTHER_ANALYZE`
- Replace with question generation prompts + investigation prompt template
- Keep PLAN and EXECUTE prompts as-is

### Step 5: Test with benchmarks
- Mega challenge (minidb 61 tests)
- Bug fix challenge (challenge2)
- Feature challenge (test2-fastapi)
- Compare: questions generated, investigation quality, plan quality, final result
