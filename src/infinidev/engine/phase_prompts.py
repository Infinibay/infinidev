"""Phase-specific prompt templates with concrete examples per task type.

Four phases: QUESTIONS → INVESTIGATE → PLAN → EXECUTE
Each task type (bug, feature, refactor, other) has tailored prompts
with multiple DO/DON'T examples.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class PhaseStrategy:
    """Configuration for how to run each phase for a given task type."""
    questions_prompt: str
    investigate_prompt: str  # per-question template
    plan_prompt: str
    execute_prompt: str
    fallback_questions: list[str] = field(default_factory=list)
    questions_min: int = 2
    questions_max: int = 10
    investigate_max_tool_calls: int = 6
    plan_min_steps: int = 3
    plan_max_step_files: int = 2
    execute_max_tool_calls_per_step: int = 8
    auto_test: bool = True
    anti_rewrite: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# INVESTIGATE prompt (shared across all task types)
# ═══════════════════════════════════════════════════════════════════════════

INVESTIGATE_PROMPT = """\
QUESTION {{q_num}}/{{q_total}}: {{question}}

Investigate this question using the available tools. When you have the answer,
save it with add_note and call step_complete.

{{previous_answers}}

## RULES
- Read files, search code, run commands — but do NOT modify any files
- You MUST call add_note with your answer BEFORE calling step_complete
- Be specific in your note: include file names, line numbers, function names
- Keep your note concise (2-4 sentences)

## EXAMPLES OF GOOD INVESTIGATION

Example 1 — Finding where code lives:
  1. search_knowledge: "authentication middleware"
     → found 1 finding: "Auth handled in src/middleware/auth.ts, uses JWT with RS256"
  2. think: "Previous knowledge says auth is in middleware/auth.ts. Let me verify and get details"
  3. read_file: "src/middleware/auth.ts" (lines 1-50)
  4. add_note: "Auth middleware at src/middleware/auth.ts:15 — verifyToken() checks JWT with RS256. Calls UserService.findById() on line 32. Throws 401 on invalid/expired token"
  5. step_complete

Example 2 — Understanding test expectations:
  1. glob: "tests/**/test_payment*" → found tests/test_payment.py, tests/test_payment_webhook.py
  2. read_file: "tests/test_payment.py" (lines 1-80)
  3. think: "Tests expect: create_charge(amount, currency) returns charge_id, get_charge(id) returns {status, amount}, refund_charge(id) returns success boolean. Uses mock Stripe client"
  4. add_note: "TESTS: test_payment.py has 15 tests. API: create_charge(amount, currency)->str, get_charge(id)->dict, refund_charge(id)->bool. Mock pattern: @patch('services.payment.stripe_client')"
  5. step_complete

Example 3 — Establishing baseline:
  1. execute_command: "cd /project && python -m pytest tests/ --tb=no -q 2>&1 | tail -5"
     → "34 passed, 12 failed in 1.2s"
  2. add_note: "BASELINE: 34/46 tests passing. 12 failures to investigate"
  3. step_complete

Example 4 — Checking existing knowledge:
  1. search_findings: "rate limiter"
     → found: "Rate limiter uses token bucket at services/rate_limit.py. Config in env: RATE_LIMIT_RPM=60"
  2. read_findings: query="rate limit"
  3. add_note: "EXISTING KNOWLEDGE: Rate limiter at services/rate_limit.py uses token bucket. Config: RATE_LIMIT_RPM env var, default 60"
  4. step_complete

## EXAMPLES OF BAD INVESTIGATION (DO NOT DO THIS)

Bad 1 — No note:
  1. read_file: "src/auth.py"
  2. step_complete: "I read the file"
  WHY BAD: No add_note. Answer is lost. Next phases have nothing to work with.

Bad 2 — Vague note:
  1. read_file: "src/models.py"
  2. add_note: "Read models.py, it has some models"
  WHY BAD: No specifics. Which models? What fields? What line numbers?

Bad 3 — Editing files:
  1. read_file: "src/handler.py"
  2. edit_file: "src/handler.py" ← WRONG
  WHY BAD: Investigation is read-only. Do NOT modify files.
"""


# ═══════════════════════════════════════════════════════════════════════════
# BUG FIX — QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════

_BUG_QUESTIONS = """\
You need to understand a bug before you can fix it. Generate questions that
will help you reproduce, locate, and understand the root cause.

## EXAMPLES OF GOOD QUESTIONS

For "Users get a 500 error when uploading files larger than 10MB":
[
  {{"question": "What error message or stack trace appears in logs when the 500 occurs?", "intent": "reproduce"}},
  {{"question": "Which function handles file uploads, and what file is it in?", "intent": "find_code"}},
  {{"question": "Is there a file size limit configured, and where is it set?", "intent": "find_config"}},
  {{"question": "Are there existing tests for file upload that cover large files?", "intent": "check_tests"}},
  {{"question": "What is the current test baseline (how many pass/fail)?", "intent": "baseline"}}
]

For "Sort order is wrong in the task list — shows oldest first instead of newest":
[
  {{"question": "Which function builds the task list query, and what ORDER BY does it use?", "intent": "find_code"}},
  {{"question": "Are there tests that verify sort order? Do they pass or fail?", "intent": "check_tests"}},
  {{"question": "Is the sort order configurable or hardcoded?", "intent": "find_config"}}
]

## EXAMPLES OF BAD QUESTIONS (DO NOT GENERATE THESE)

- "What is this project?" (too broad, not relevant to the bug)
- "What programming language is used?" (obvious from the code)
- "Can you explain the architecture?" (not targeted — ask about the specific area)
- "What should I fix?" (the task already tells you)
"""

_BUG_FALLBACK = [
    "What error message, stack trace, or failing test reproduces this bug?",
    "What function and file contains the buggy code?",
    "Are there existing tests that cover this behavior, and do they pass or fail?",
]


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE — QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════

_FEATURE_QUESTIONS = """\
You need to understand the codebase and the specification before implementing.
Generate questions that will reveal the API, patterns, dependencies, and current state.

## EXAMPLES OF GOOD QUESTIONS

For "Add a /users/stats endpoint that returns activity metrics":
[
  {{"question": "What patterns do existing endpoints follow? (decorators, auth, response format)", "intent": "find_patterns"}},
  {{"question": "What data models exist for users and their activity?", "intent": "find_code"}},
  {{"question": "Are there tests for the new endpoint already written? What do they expect?", "intent": "check_tests"}},
  {{"question": "What is the current test baseline?", "intent": "baseline"}},
  {{"question": "Are there similar stats/aggregation endpoints I can use as reference?", "intent": "find_patterns"}}
]

For "Implement a caching layer for database queries":
[
  {{"question": "How are database queries currently made? (ORM, raw SQL, query builder)", "intent": "find_code"}},
  {{"question": "What existing caching infrastructure exists? (Redis, in-memory, none)", "intent": "find_code"}},
  {{"question": "Which queries are most frequently called and would benefit from caching?", "intent": "find_code"}},
  {{"question": "Are there tests that verify query results? Will caching change behavior?", "intent": "check_tests"}},
  {{"question": "What is the project structure and where should new modules go?", "intent": "find_patterns"}}
]

For "Build a module from scratch based on a test specification":
[
  {{"question": "What is the full public API expected? (classes, methods, signatures, return types)", "intent": "understand_spec"}},
  {{"question": "How many tests are there and what categories do they cover?", "intent": "check_tests"}},
  {{"question": "What are the dependencies between features? (what must be built first)", "intent": "understand_spec"}},
  {{"question": "Are there existing files or patterns in the project to follow?", "intent": "find_patterns"}},
  {{"question": "What is the current test baseline?", "intent": "baseline"}}
]

## EXAMPLES OF BAD QUESTIONS (DO NOT GENERATE THESE)

- "What should I implement?" (the task already says this)
- "Is there documentation?" (go read the files instead of asking)
- "How does everything work?" (too broad — ask about specific areas)
- "What tests should I write?" (read the test file to find out)
"""

_FEATURE_FALLBACK = [
    "What is the expected API and behavior? (read tests/specs if they exist)",
    "What existing code patterns should the implementation follow?",
    "What is the current test baseline?",
    "What are the dependencies between components? (build order)",
]


# ═══════════════════════════════════════════════════════════════════════════
# REFACTOR — QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════

_REFACTOR_QUESTIONS = """\
Before refactoring, you must understand the code AND all its consumers.
Generate questions that map dependencies and establish a safety baseline.

## EXAMPLES OF GOOD QUESTIONS

For "Split the monolithic process() function into smaller helpers":
[
  {{"question": "What does process() do and what are its logical sections?", "intent": "find_code"}},
  {{"question": "Who calls process()? (all importers and callers across the project)", "intent": "find_dependents"}},
  {{"question": "What is the current test baseline? All tests must keep passing", "intent": "baseline"}},
  {{"question": "Are there internal variables shared across the sections that complicate extraction?", "intent": "find_code"}}
]

For "Move UserService from services/ to a new domain/ directory":
[
  {{"question": "What files import UserService? (all references across the project)", "intent": "find_dependents"}},
  {{"question": "Does UserService depend on other services that would also need to move?", "intent": "find_code"}},
  {{"question": "What is the current test baseline?", "intent": "baseline"}}
]

## EXAMPLES OF BAD QUESTIONS (DO NOT GENERATE THESE)

- "What does the code do?" (too vague — ask about the specific function/class)
- "Should I refactor this?" (the task already asks you to)
"""

_REFACTOR_FALLBACK = [
    "What is the full test baseline? (must not regress after refactoring)",
    "What files and functions need to change?",
    "Who calls/imports the code being refactored? (all dependents)",
]


# ═══════════════════════════════════════════════════════════════════════════
# OTHER / SYSADMIN — QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════

_OTHER_QUESTIONS = """\
Generate questions to understand the current state before making changes.

## EXAMPLES OF GOOD QUESTIONS

For "Change the API timeout from 30s to 60s":
[
  {{"question": "Where is the timeout configured? (file, line, env var)", "intent": "find_config"}},
  {{"question": "Are there other timeout settings that might need to change too?", "intent": "find_config"}}
]

For "Figure out why the deploy is failing":
[
  {{"question": "What does the deploy error log say?", "intent": "reproduce"}},
  {{"question": "What changed recently that could cause the failure?", "intent": "find_code"}},
  {{"question": "What is the deploy process? (scripts, CI config, commands)", "intent": "find_code"}}
]

## EXAMPLES OF BAD QUESTIONS

- "What is the project?" (too broad)
"""

_OTHER_FALLBACK = [
    "What is the current state of the system/config related to the task?",
    "What needs to change and where?",
]


# ═══════════════════════════════════════════════════════════════════════════
# PLAN prompts (same as before, kept here for reference)
# ═══════════════════════════════════════════════════════════════════════════

_BUG_PLAN = """\
Create a step-by-step plan to fix the bug(s) found in your investigation.

## RULES
- Each step fixes ONE specific thing in ONE function
- After each fix, include a step to run the specific failing test
- Do NOT fix things unrelated to the bug
- Do NOT rewrite entire files

## EXAMPLES OF GOOD PLANS

Example 1 — Two bugs in one file:
[
  {{"step": 1, "description": "Fix verify_token() in auth.py:52 — add expiry check: if payload['exp'] < time.time() return None", "files": ["src/auth.py"]}},
  {{"step": 2, "description": "Run pytest tests/test_auth.py::test_expired_token to verify fix", "files": []}},
  {{"step": 3, "description": "Fix refresh_token() in auth.py:85 — replace 'return None' with 'return create_token(user_id)'", "files": ["src/auth.py"]}},
  {{"step": 4, "description": "Run pytest tests/test_auth.py -v to verify all tests pass", "files": []}}
]

Example 2 — Bug with missing test:
[
  {{"step": 1, "description": "Add None check in get_user_display_name() at models.py:123 — return 'Unknown' if name is None", "files": ["models.py"]}},
  {{"step": 2, "description": "Run existing tests: pytest tests/test_models.py -v", "files": []}},
  {{"step": 3, "description": "Add test_display_name_none_user() to tests/test_models.py — verify None name returns 'Unknown'", "files": ["tests/test_models.py"]}},
  {{"step": 4, "description": "Run full test suite to verify no regressions", "files": []}}
]

## EXAMPLES OF BAD PLANS (DO NOT DO THIS)

Bad:
[
  {{"step": 1, "description": "Fix the authentication bugs", "files": ["src/auth.py"]}},
  {{"step": 2, "description": "Run tests", "files": []}}
]
WHY BAD: Step 1 is vague — which bugs? which functions? Two different fixes should be two steps.
"""


_FEATURE_PLAN = """\
Create a step-by-step implementation plan. Build incrementally — each step adds one capability.

## RULES
- First step: create file skeleton with stubs (if new file) or add method stub (if extending)
- Each step implements ONE method or ONE small group of tightly related methods
- Each step should make progress visible (more tests pass, or a sub-feature works)
- Include a "run tests and check progress" step after every 2-3 implementation steps
- Order by dependency: foundations first, complex features last
- Use edit_file to ADD code to existing files, not write_file to REPLACE them
- Max 2 files per step

## EXAMPLES OF GOOD PLANS

Example 1 — Adding API endpoints:
[
  {{"step": 1, "description": "Add GET /reports route in routes/reports.py with empty response skeleton, register in app", "files": ["routes/reports.py", "app.py"]}},
  {{"step": 2, "description": "Implement report generation logic: query orders by date range, aggregate by category", "files": ["routes/reports.py"]}},
  {{"step": 3, "description": "Add pagination support: accept page/per_page params, return metadata", "files": ["routes/reports.py"]}},
  {{"step": 4, "description": "Run tests: pytest tests/test_reports.py -v", "files": []}}
]

Example 2 — Building a module incrementally:
[
  {{"step": 1, "description": "Create parser.py with Parser class skeleton: __init__(), parse() stub, Token dataclass", "files": ["parser.py"]}},
  {{"step": 2, "description": "Implement tokenizer: split input into Token objects (NUMBER, STRING, KEYWORD, OPERATOR)", "files": ["parser.py"]}},
  {{"step": 3, "description": "Implement expression parsing: handle binary ops (+, -, *, /) with precedence", "files": ["parser.py"]}},
  {{"step": 4, "description": "Run tests to check basic parsing works", "files": []}},
  {{"step": 5, "description": "Add function call parsing: detect name(...args) pattern, build call nodes", "files": ["parser.py"]}},
  {{"step": 6, "description": "Add error handling: raise ParseError with line/column info for invalid input", "files": ["parser.py"]}},
  {{"step": 7, "description": "Run full test suite", "files": []}}
]

## EXAMPLES OF BAD PLANS (DO NOT DO THIS)

Bad 1 — Too vague:
[
  {{"step": 1, "description": "Implement the parser", "files": ["parser.py"]}},
  {{"step": 2, "description": "Test it", "files": []}}
]
WHY BAD: "Implement the parser" is hundreds of lines. One step should be one method or one feature.

Bad 2 — No test steps:
[
  {{"step": 1, "description": "Add tokenizer", "files": ["parser.py"]}},
  {{"step": 2, "description": "Add expression parsing", "files": ["parser.py"]}},
  {{"step": 3, "description": "Add function calls", "files": ["parser.py"]}},
  ... (no testing anywhere) ...
]
WHY BAD: Never checks progress. Building on a broken foundation.

Bad 3 — Wrong order:
[
  {{"step": 1, "description": "Implement function call parsing", "files": ["parser.py"]}},
  {{"step": 2, "description": "Implement tokenizer", "files": ["parser.py"]}}
]
WHY BAD: Function calls depend on tokens existing. Build foundations first.
"""


_REFACTOR_PLAN = """\
Create a plan for the refactoring. Each step is ONE atomic change. Tests MUST pass after EVERY step.

## RULES
- Never change behavior and structure in the same step
- Include "run full test suite" after EVERY step (not just related tests)
- If renaming: one step to add the new name, one to update callers, one to remove old
- If extracting: one step per extracted function/class

## EXAMPLES OF GOOD PLANS

Example — Extracting helpers:
[
  {{"step": 1, "description": "Extract _validate_input() from handle_request() lines 20-45 in handler.py — move validation logic to new function, call from handle_request()", "files": ["src/handler.py"]}},
  {{"step": 2, "description": "Run pytest tests/ -v to verify no regressions", "files": []}},
  {{"step": 3, "description": "Extract _format_response() from handle_request() lines 80-110 in handler.py", "files": ["src/handler.py"]}},
  {{"step": 4, "description": "Run full test suite to verify all tests still pass", "files": []}}
]

## EXAMPLES OF BAD PLANS (DO NOT DO THIS)

Bad:
[
  {{"step": 1, "description": "Refactor the module", "files": ["src/handler.py"]}},
  {{"step": 2, "description": "Run tests", "files": []}}
]
WHY BAD: "Refactor" is not a step. Each extraction/rename is its own step with tests between.
"""


_OTHER_PLAN = """\
Create a simple plan for the task.

## RULES
- Each step is one specific action
- Include verification after each change
- Be specific about files and what changes

## EXAMPLE
[
  {{"step": 1, "description": "Change timeout from 30 to 60 in config/settings.yaml line 15", "files": ["config/settings.yaml"]}},
  {{"step": 2, "description": "Verify change: grep timeout config/settings.yaml", "files": []}}
]
"""


# ═══════════════════════════════════════════════════════════════════════════
# EXECUTE prompts
# ═══════════════════════════════════════════════════════════════════════════

_BUG_EXECUTE = """\
STEP {{step_num}}/{{total_steps}}: {{step_description}}
Files you may modify: {{step_files}}

## RULES
- ONLY modify the file(s) and function(s) described in this step
- Use edit_file with the smallest possible change
- Do NOT rewrite the whole function — change only the broken line(s)
- Verify your edit: run the relevant test or a quick syntax check
- Call step_complete when done

## EXAMPLES OF GOOD EXECUTION

Example — Fixing one comparison:
  1. edit_file: path="src/auth.py", old_string="if payload['exp']:", new_string="if payload.get('exp', 0) < time.time():"
  2. execute_command: "pytest tests/test_auth.py::test_expired_token -v"
     → PASSED
  3. step_complete: summary="Fixed expiry check in verify_token(). Test passes."

Example — Adding a guard clause:
  1. read_file: "models.py" (lines 120-125 to see exact current code)
  2. edit_file: path="models.py", old_string="return name.lower()", new_string="return name.lower() if name else 'Unknown'"
  3. execute_command: "python -c 'from models import User; print(User(name=None).display_name())'"
     → "Unknown"
  4. step_complete: summary="Added None guard for name field."

## EXAMPLES OF BAD EXECUTION (DO NOT DO THIS)

Bad — Rewriting the whole file:
  write_file: path="src/auth.py", content="(entire 200-line file rewritten)"
  WHY BAD: Rewrites everything to fix one line. Use edit_file.

Bad — Fixing things not in this step:
  Step says "Fix verify_token()" but you also edit refresh_token()
  WHY BAD: Stay in scope. Other fixes go in their own step.
"""


_FEATURE_EXECUTE = """\
STEP {{step_num}}/{{total_steps}}: {{step_description}}
Files you may modify: {{step_files}}

## RULES
- ONLY implement what this step describes — nothing more
- If creating a new file: use write_file (keep it short, just this step's code)
- If modifying existing file: use edit_file (old_string → new_string)
- Do NOT rewrite entire files to add a function — use edit_file to insert
- After editing, verify: run a quick import check or the relevant tests
- Call step_complete with a summary of what you added/changed

## EXAMPLES OF GOOD EXECUTION

Example — Creating a file skeleton:
  1. write_file: path="src/cache.py", content="class Cache:\\n    def __init__(self):\\n        self._store = {{}}\\n\\n    def get(self, key):\\n        pass  # TODO\\n\\n    def set(self, key, value):\\n        pass  # TODO\\n"
  2. execute_command: "python -c 'from src.cache import Cache; c = Cache(); print(type(c))'"
     → <class 'src.cache.Cache'>
  3. step_complete: "Created Cache class skeleton with get/set stubs"

Example — Adding a method to existing file:
  1. read_file: "src/cache.py" (last 5 lines to find insertion point)
  2. edit_file: path="src/cache.py",
     old_string="    def set(self, key, value):\\n        pass  # TODO",
     new_string="    def set(self, key, value, ttl=None):\\n        self._store[key] = {{'value': value, 'expires': time.time() + ttl if ttl else None}}\\n\\n    def get(self, key):\\n        entry = self._store.get(key)\\n        if not entry: return None\\n        if entry['expires'] and time.time() > entry['expires']:\\n            del self._store[key]\\n            return None\\n        return entry['value']"
  3. execute_command: "python -c 'from src.cache import Cache; c = Cache(); c.set(\"k\", 42); print(c.get(\"k\"))'"
     → 42
  4. step_complete: "Implemented get/set with TTL expiration"

Example — Running tests (verification step):
  1. execute_command: "pytest tests/test_cache.py --tb=short -q 2>&1 | tail -10"
     → "8 passed, 4 failed"
  2. step_complete: "Progress: 8/12 tests passing (up from 3)"

## EXAMPLES OF BAD EXECUTION (DO NOT DO THIS)

Bad — Rewriting entire file:
  Step says "Add TTL support to Cache.set()"
  write_file: path="src/cache.py", content="(entire 300-line file)"
  WHY BAD: Overwrites everything. Previous working code might break. Use edit_file.

Bad — Going beyond the step:
  Step says "Add set() method" but you also add delete(), clear(), stats()
  WHY BAD: One step = one thing. Other methods go in their own steps.

Bad — No verification:
  1. edit_file: (changes code)
  2. step_complete: "Done"
  WHY BAD: Didn't verify. Always check for syntax errors or run a quick test.
"""


_REFACTOR_EXECUTE = """\
STEP {{step_num}}/{{total_steps}}: {{step_description}}
Files you may modify: {{step_files}}

## RULES
- Make ONE structural change per step
- Use edit_file — do NOT rewrite the entire file
- After editing, ALWAYS run the full test suite (not just one test)
- If any test breaks: undo your change and rethink
- Call step_complete with what changed and test results

## EXAMPLES OF GOOD EXECUTION

Example — Extracting a function:
  1. read_file: "src/handler.py" (lines 20-45 to see exact code)
  2. edit_file: path="src/handler.py",
     old_string="    # Validate input\\n    if not data.get('name'):\\n        raise ValueError('name required')\\n    if len(data['name']) > 100:\\n        raise ValueError('name too long')",
     new_string="    _validate_input(data)"
  3. edit_file: path="src/handler.py",
     old_string="def handle_request(data):",
     new_string="def _validate_input(data):\\n    if not data.get('name'):\\n        raise ValueError('name required')\\n    if len(data['name']) > 100:\\n        raise ValueError('name too long')\\n\\n\\ndef handle_request(data):"
  4. execute_command: "pytest tests/ -q"
     → "48 passed"
  5. step_complete: "Extracted _validate_input(). All 48 tests pass."

## EXAMPLES OF BAD EXECUTION (DO NOT DO THIS)

Bad:
  write_file: path="src/handler.py", content="(completely rewritten file)"
  WHY BAD: Rewrote everything for one extraction. High regression risk.
"""


_OTHER_EXECUTE = """\
STEP {{step_num}}/{{total_steps}}: {{step_description}}
Files you may modify: {{step_files}}

## RULES
- Do exactly what the step says
- Verify the change took effect
- Call step_complete when done

## EXAMPLE
  1. edit_file: path="config/settings.yaml", old_string="timeout: 30", new_string="timeout: 60"
  2. execute_command: "grep timeout config/settings.yaml"
     → "timeout: 60"
  3. step_complete: "Changed timeout from 30 to 60. Verified."
"""


# ═══════════════════════════════════════════════════════════════════════════
# Strategy registry
# ═══════════════════════════════════════════════════════════════════════════

STRATEGIES: dict[str, PhaseStrategy] = {
    "bug": PhaseStrategy(
        questions_prompt=_BUG_QUESTIONS,
        investigate_prompt=INVESTIGATE_PROMPT,
        plan_prompt=_BUG_PLAN,
        execute_prompt=_BUG_EXECUTE,
        fallback_questions=_BUG_FALLBACK,
        questions_min=2,
        questions_max=6,
        investigate_max_tool_calls=12,
        plan_min_steps=2,
        plan_max_step_files=2,
        execute_max_tool_calls_per_step=6,
        auto_test=True,
    ),
    "feature": PhaseStrategy(
        questions_prompt=_FEATURE_QUESTIONS,
        investigate_prompt=INVESTIGATE_PROMPT,
        plan_prompt=_FEATURE_PLAN,
        execute_prompt=_FEATURE_EXECUTE,
        fallback_questions=_FEATURE_FALLBACK,
        questions_min=3,
        questions_max=10,
        investigate_max_tool_calls=12,
        plan_min_steps=4,
        plan_max_step_files=2,
        execute_max_tool_calls_per_step=8,
        auto_test=True,
        anti_rewrite=True,
    ),
    "refactor": PhaseStrategy(
        questions_prompt=_REFACTOR_QUESTIONS,
        investigate_prompt=INVESTIGATE_PROMPT,
        plan_prompt=_REFACTOR_PLAN,
        execute_prompt=_REFACTOR_EXECUTE,
        fallback_questions=_REFACTOR_FALLBACK,
        questions_min=2,
        questions_max=6,
        investigate_max_tool_calls=12,
        plan_min_steps=3,
        plan_max_step_files=3,
        execute_max_tool_calls_per_step=8,
        auto_test=True,
        anti_rewrite=True,
    ),
    "other": PhaseStrategy(
        questions_prompt=_OTHER_QUESTIONS,
        investigate_prompt=INVESTIGATE_PROMPT,
        plan_prompt=_OTHER_PLAN,
        execute_prompt=_OTHER_EXECUTE,
        fallback_questions=_OTHER_FALLBACK,
        questions_min=1,
        questions_max=5,
        investigate_max_tool_calls=12,
        plan_min_steps=1,
        plan_max_step_files=3,
        execute_max_tool_calls_per_step=8,
    ),
    "sysadmin": PhaseStrategy(
        questions_prompt=_OTHER_QUESTIONS,
        investigate_prompt=INVESTIGATE_PROMPT,
        plan_prompt=_OTHER_PLAN,
        execute_prompt=_OTHER_EXECUTE,
        fallback_questions=_OTHER_FALLBACK,
        questions_min=1,
        questions_max=5,
        investigate_max_tool_calls=12,
        plan_min_steps=1,
        plan_max_step_files=3,
        execute_max_tool_calls_per_step=8,
    ),
}


def get_strategy(task_type: str) -> PhaseStrategy:
    """Get the phase strategy for a task type. Defaults to 'feature' for unknown types."""
    return STRATEGIES.get(task_type, STRATEGIES["feature"])
