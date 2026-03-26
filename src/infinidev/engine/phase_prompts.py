"""Phase-specific prompt templates with concrete examples per task type.

Each task type (bug, feature, refactor, implement, other) has tailored
prompts for ANALYZE, PLAN, and EXECUTE phases with multiple DO/DON'T
examples showing exactly what good and bad behavior looks like.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class PhaseStrategy:
    """Configuration for how to run each phase for a given task type."""
    analyze_prompt: str
    plan_prompt: str
    execute_prompt: str
    analyze_max_tool_calls: int = 15
    plan_min_steps: int = 3
    plan_max_step_files: int = 2
    execute_max_tool_calls_per_step: int = 8
    auto_test: bool = True
    anti_rewrite: bool = False  # warn if write_file on existing file


# ═══════════════════════════════════════════════════════════════════════════
# BUG FIX
# ═══════════════════════════════════════════════════════════════════════════

_BUG_ANALYZE = """\
You are in ANALYSIS MODE. Your goal: reproduce the bug and locate the root cause.

DO NOT write or edit any files. Only read, search, and run commands.
You MUST use add_note after each significant finding.

## EXAMPLES OF GOOD ANALYSIS

Example 1 — Failing test:
  1. execute_command: "pytest tests/test_auth.py -v --tb=short"
     → 2 tests fail: test_expired_token, test_refresh_returns_none
  2. add_note: "FAILING: test_expired_token expects 401 but gets 200. test_refresh_returns_none expects new token but gets None"
  3. read_file: "src/auth.py" (lines 40-90)
  4. add_note: "auth.py:52 verify_token() — missing expiry check. Compares signature but never checks 'exp' field against current time"
  5. add_note: "auth.py:78 refresh_token() — returns None on line 85 instead of generating new token. The else branch is wrong"
  6. phase_complete: "Root cause found: 2 bugs in auth.py"

Example 2 — Error traceback:
  1. read_file: "error_log.txt" (or reproduce with execute_command)
  2. add_note: "ERROR: TypeError at models.py:123 — calling .lower() on None. The 'name' field can be None but code assumes string"
  3. code_search: "def get_user" → found in models.py:100
  4. read_file: "models.py" (lines 100-130)
  5. add_note: "models.py:123 get_user_display_name() — does name.lower() but name can be NULL in DB (nullable column)"
  6. glob: "tests/**/test_models*"
  7. read_file: "tests/test_models.py" → check if test exists for None name
  8. add_note: "No test for None name case. Need to add one after fixing"
  9. phase_complete: "Bug: None check missing in get_user_display_name"

Example 3 — Behavior mismatch:
  1. execute_command: "python -c 'from mylib import calc; print(calc.discount(100, 20))'"
     → outputs: -1800 (expected: 80)
  2. add_note: "REPRO: discount(100, 20) returns -1800. Expected 80 (100 * 0.80)"
  3. read_file: "mylib/calc.py"
  4. add_note: "calc.py:15 discount() — uses price * (1 - percent) but percent=20 not 0.20. Should divide by 100 first"
  5. phase_complete: "Root cause: percent not converted to fraction"

## EXAMPLES OF BAD ANALYSIS (DO NOT DO THIS)

Bad 1 — No notes, no reproduction:
  1. read_file: "src/auth.py"
  2. read_file: "src/models.py"
  3. read_file: "src/views.py"
  4. phase_complete: "I've read the code and understand the issue"
  WHY BAD: No notes taken, no test run, no specific findings. The next phase has nothing to work with.

Bad 2 — Jumping to fixes:
  1. execute_command: "pytest" → sees failures
  2. edit_file: "src/auth.py" ← WRONG: analysis phase is read-only
  WHY BAD: Editing before understanding. Might fix symptom but miss root cause.

Bad 3 — Reading everything:
  1. read_file: "README.md"
  2. read_file: "setup.py"
  3. read_file: "requirements.txt"
  4. list_directory: "."
  5. list_directory: "src/"
  ... (wastes all tool calls reading irrelevant files)
  WHY BAD: Not targeted. Start with the error/test, trace from there.
"""


_BUG_PLAN = """\
Create a step-by-step plan to fix the bug(s) found in analysis.

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
  3. step_complete: summary="Fixed expiry check in verify_token(). Test passes now."

Example — Adding a None guard:
  1. read_file: "models.py" (lines 120-125 to see exact current code)
  2. edit_file: path="models.py", old_string="return name.lower()", new_string="return name.lower() if name else 'Unknown'"
  3. execute_command: "python -c 'from models import get_user_display_name; print(get_user_display_name(None))'"
     → "Unknown"
  4. step_complete: summary="Added None guard. Returns 'Unknown' for null names."

## EXAMPLES OF BAD EXECUTION (DO NOT DO THIS)

Bad — Rewriting the whole function:
  write_file: path="src/auth.py", content="(entire 200-line file rewritten)"
  WHY BAD: Rewrites everything to fix one line. Use edit_file with old_string/new_string.

Bad — Fixing things not in this step:
  Step says "Fix verify_token()" but you also edit refresh_token()
  WHY BAD: Stay in scope. Other fixes go in their own step.
"""


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE / IMPLEMENT FROM SCRATCH
# ═══════════════════════════════════════════════════════════════════════════

_FEATURE_ANALYZE = """\
You are in ANALYSIS MODE. Your goal: understand existing code patterns and where the new feature integrates.

DO NOT write or edit any files. Only read, search, and run commands.
You MUST use add_note after each significant finding.

## EXAMPLES OF GOOD ANALYSIS

Example 1 — Adding a new API endpoint:
  1. glob: "app/routes/*.py" → find existing route files
  2. read_file: "app/routes/users.py" → see pattern for existing endpoints
  3. add_note: "PATTERN: routes use @app.get/post decorators, return Pydantic models, auth via Depends(get_current_user)"
  4. read_file: "app/models.py" → check existing data models
  5. add_note: "MODELS: User(id, name, email), Task(id, title, owner_id). Need to add Stats model or return dict"
  6. read_file: "tests/test_users.py" → see test patterns
  7. add_note: "TEST PATTERN: uses TestClient, fixtures create users with auth headers, asserts on response.json()"
  8. execute_command: "pytest tests/ --tb=no -q" → baseline
  9. add_note: "BASELINE: 16 pass, 0 fail. New endpoint tests will start failing"
  10. phase_complete: "Understand patterns. Ready to plan stats endpoint."

Example 2 — Implementing a module from a test spec:
  1. read_file: "test_minidb.py" (lines 1-100)
  2. add_note: "API: Database class with execute(sql), get_tables(), get_schema(table), export_csv(table, path), import_csv(table, path)"
  3. read_file: "test_minidb.py" (lines 100-200)
  4. add_note: "CREATE TABLE: supports INTEGER, TEXT, FLOAT, BOOLEAN columns. PRIMARY KEY with auto-increment. NOT NULL and UNIQUE constraints"
  5. read_file: "test_minidb.py" (lines 200-300)
  6. add_note: "SELECT: needs WHERE (=, !=, >, <, AND, OR, LIKE, BETWEEN, IN, IS NULL), ORDER BY, LIMIT, OFFSET, DISTINCT, aliases"
  7. read_file: "test_minidb.py" (lines 300-400)
  8. add_note: "JOINS: INNER JOIN and LEFT JOIN on column equality. Aggregates: COUNT, SUM, AVG, MIN, MAX. GROUP BY with HAVING"
  9. read_file: "test_minidb.py" (lines 400-end)
  10. add_note: "ALSO NEEDED: UPDATE, DELETE, ALTER TABLE, transactions (BEGIN/COMMIT/ROLLBACK), subqueries, CSV import/export, foreign keys with CASCADE"
  11. add_note: "DEPENDENCY ORDER: Table storage → CREATE/INSERT → basic SELECT → WHERE → ORDER/LIMIT → UPDATE/DELETE → JOINs → aggregates → subqueries → transactions → CSV → constraints"
  12. execute_command: "pytest test_minidb.py --tb=no -q 2>&1 | tail -3"
     → ERROR (file doesn't exist yet)
  13. add_note: "BASELINE: 0/61 tests. File needs to be created from scratch"
  14. phase_complete: "Full spec analyzed. 61 tests, 12 feature areas identified."

Example 3 — Extending existing feature:
  1. code_search: "class ShortenerService" → found in shortener/service.py
  2. read_file: "shortener/service.py"
  3. add_note: "ShortenerService has: shorten(), resolve(), get_stats(), list_urls(), delete_url(). Need to add expiration"
  4. read_file: "shortener/storage.py"
  5. add_note: "URLStore.store() creates entry with url, created_at, clicks[]. Need to add expires_at field"
  6. read_file: "test_shortener.py" → find the expiration tests
  7. add_note: "4 expiration tests: test_shorten_with_expiry, test_expired_url_not_resolved, test_non_expired_url_resolves, test_expired_url_still_has_stats"
  8. add_note: "Tests mock datetime via patch('shortener.service.datetime') — resolve() must use module-level datetime, not stored reference"
  9. phase_complete: "Need to modify service.py and storage.py. Mock pattern requires careful datetime handling."

## EXAMPLES OF BAD ANALYSIS (DO NOT DO THIS)

Bad 1 — Not reading the spec:
  1. list_directory: "."
  2. phase_complete: "Ready to implement"
  WHY BAD: Didn't read tests or existing code. Plan will be vague.

Bad 2 — Reading without noting:
  1. read_file: "test_minidb.py"
  2. read_file: "test_minidb.py" (again, different section)
  3. read_file: "test_minidb.py" (again)
  4. phase_complete: "I understand the tests"
  WHY BAD: No add_note calls. All findings are lost between phases.
"""


_FEATURE_PLAN = """\
Create a step-by-step implementation plan. Build incrementally — each step adds one capability.

## RULES
- First step: create file skeleton with stubs (if new file) or add method stub (if extending)
- Each step implements ONE method or ONE small group of tightly related methods
- Each step should make 2-5 more tests pass (for test-driven tasks)
- Include a "run tests and check progress" step after every 2-3 implementation steps
- Order by dependency: foundations first, complex features last
- Use edit_file to ADD code to existing files, not write_file to REPLACE them
- Max 2 files per step

## EXAMPLES OF GOOD PLANS

Example 1 — Implementing from test spec (minidb):
[
  {{"step": 1, "description": "Create minidb.py with Database class skeleton: __init__() with table storage dict, execute() that dispatches by SQL command, get_tables(), get_schema()", "files": ["minidb.py"]}},
  {{"step": 2, "description": "Implement CREATE TABLE parsing in execute() — regex to extract table name and column definitions, store in self.tables dict", "files": ["minidb.py"]}},
  {{"step": 3, "description": "Implement INSERT INTO — parse column names and values, add row to table, handle auto-increment PRIMARY KEY", "files": ["minidb.py"]}},
  {{"step": 4, "description": "Run tests: pytest test_minidb.py -v --tb=short 2>&1 | head -40", "files": []}},
  {{"step": 5, "description": "Implement basic SELECT — parse column list (or *), return rows from table as list of tuples", "files": ["minidb.py"]}},
  {{"step": 6, "description": "Add WHERE clause evaluation — parse conditions (=, !=, >, <, >=, <=), filter rows", "files": ["minidb.py"]}},
  {{"step": 7, "description": "Add AND/OR logic to WHERE parser, add LIKE with % wildcard, add BETWEEN and IN operators", "files": ["minidb.py"]}},
  {{"step": 8, "description": "Run tests to check progress on TestSelect and TestWhere", "files": []}},
  {{"step": 9, "description": "Implement ORDER BY (ASC/DESC, multiple columns), LIMIT, OFFSET", "files": ["minidb.py"]}},
  {{"step": 10, "description": "Implement UPDATE with SET and WHERE, DELETE with WHERE", "files": ["minidb.py"]}},
  {{"step": 11, "description": "Run tests to check TestOrderLimit, TestUpdateDelete", "files": []}},
  {{"step": 12, "description": "Implement INNER JOIN and LEFT JOIN — parse ON condition, combine rows from two tables", "files": ["minidb.py"]}},
  {{"step": 13, "description": "Implement aggregate functions: COUNT, SUM, AVG, MIN, MAX. Add GROUP BY with HAVING", "files": ["minidb.py"]}},
  {{"step": 14, "description": "Run tests to check TestJoins and TestAggregates", "files": []}},
  {{"step": 15, "description": "Implement DISTINCT, column aliases (AS), NOT NULL and UNIQUE constraints", "files": ["minidb.py"]}},
  {{"step": 16, "description": "Implement ALTER TABLE (ADD COLUMN, DROP COLUMN)", "files": ["minidb.py"]}},
  {{"step": 17, "description": "Implement transactions: BEGIN saves snapshot, COMMIT clears it, ROLLBACK restores it", "files": ["minidb.py"]}},
  {{"step": 18, "description": "Run tests to check TestAlterTable, TestTransactions, TestEdgeCases", "files": []}},
  {{"step": 19, "description": "Implement subqueries in WHERE clause — detect nested SELECT, evaluate inner query first", "files": ["minidb.py"]}},
  {{"step": 20, "description": "Implement CSV export_csv() and import_csv() methods", "files": ["minidb.py"]}},
  {{"step": 21, "description": "Implement FOREIGN KEY constraint with ON DELETE CASCADE, CREATE INDEX", "files": ["minidb.py"]}},
  {{"step": 22, "description": "Run full test suite: pytest test_minidb.py -v — aim for 61/61", "files": []}}
]

Example 2 — Adding an endpoint:
[
  {{"step": 1, "description": "Add GET /tasks/stats route in app/main.py BEFORE /tasks/{{task_id}} to avoid path conflict. Return empty stats dict for now", "files": ["app/main.py"]}},
  {{"step": 2, "description": "Implement stats calculation: count total, completed, pending, overdue tasks from get_tasks_by_owner()", "files": ["app/main.py"]}},
  {{"step": 3, "description": "Add by_priority breakdown to stats — count pending tasks per priority level (1, 2, 3)", "files": ["app/main.py"]}},
  {{"step": 4, "description": "Run pytest test_api.py -v to verify all TestTaskStats pass", "files": []}}
]

## EXAMPLES OF BAD PLANS (DO NOT DO THIS)

Bad 1 — Too vague:
[
  {{"step": 1, "description": "Implement the database", "files": ["minidb.py"]}},
  {{"step": 2, "description": "Test it", "files": []}}
]
WHY BAD: "Implement the database" is 1000 lines. One step should be one method.

Bad 2 — No test steps:
[
  {{"step": 1, "description": "Add CREATE TABLE", "files": ["minidb.py"]}},
  {{"step": 2, "description": "Add INSERT", "files": ["minidb.py"]}},
  {{"step": 3, "description": "Add SELECT", "files": ["minidb.py"]}},
  ... 15 more steps with no testing ...
]
WHY BAD: Never checks progress. Could be building on broken foundation.

Bad 3 — Wrong order:
[
  {{"step": 1, "description": "Implement JOINs", "files": ["minidb.py"]}},
  {{"step": 2, "description": "Implement CREATE TABLE", "files": ["minidb.py"]}}
]
WHY BAD: JOINs depend on tables existing. Build foundations first.
"""


_FEATURE_EXECUTE = """\
STEP {{step_num}}/{{total_steps}}: {{step_description}}
Files you may modify: {{step_files}}

## RULES
- ONLY implement what this step describes — nothing more
- If creating a new file: use write_file
- If modifying existing file: use edit_file (old_string → new_string)
- Do NOT rewrite the entire file to add a function — add it with edit_file
- After editing, do a quick verify: python -c "import module_name"
- Call step_complete with a summary of what you added/changed

## EXAMPLES OF GOOD EXECUTION

Example — Creating file skeleton (step 1 of new implementation):
  1. write_file: path="minidb.py", content="class Database:\\n    def __init__(self):\\n        self.tables = {{}}\\n..."
     (just the skeleton, ~30-50 lines with stub methods)
  2. execute_command: "python -c 'from minidb import Database; db = Database(); print(db.get_tables())'"
     → [] (works)
  3. step_complete: "Created Database class skeleton with stubs for execute, get_tables, get_schema, export_csv, import_csv"

Example — Adding one method to existing file:
  1. read_file: "minidb.py" (last 10 lines to find insertion point)
  2. edit_file: path="minidb.py",
     old_string="    def get_schema(self, table_name):",
     new_string="    def _execute_insert(self, sql):\\n        # parse INSERT INTO table (cols) VALUES (vals)\\n        ...\\n\\n    def get_schema(self, table_name):"
  3. execute_command: "python -c 'from minidb import Database'"  → no syntax error
  4. step_complete: "Added _execute_insert() method. Handles column names, values, auto-increment, type coercion"

Example — Running tests (verification step):
  1. execute_command: "pytest test_minidb.py --tb=short -q 2>&1 | tail -10"
     → "15 passed, 46 failed in 0.3s"
  2. step_complete: "Progress: 15/61 tests passing (up from 8 last check)"

## EXAMPLES OF BAD EXECUTION (DO NOT DO THIS)

Bad — Rewriting entire file:
  Step says "Add WHERE clause to SELECT"
  write_file: path="minidb.py", content="(entire 500-line file with WHERE added)"
  WHY BAD: Rewrites everything. Previous working code might break. Use edit_file to add the WHERE logic.

Bad — Doing more than the step says:
  Step says "Add WHERE clause" but you also add ORDER BY, LIMIT, GROUP BY
  WHY BAD: One step = one thing. Keep it focused. Other features go in their own steps.

Bad — No verification:
  1. edit_file: (makes changes)
  2. step_complete: "Done"
  WHY BAD: Didn't check for syntax errors or basic functionality. Always verify.
"""


# ═══════════════════════════════════════════════════════════════════════════
# REFACTOR
# ═══════════════════════════════════════════════════════════════════════════

_REFACTOR_ANALYZE = """\
You are in ANALYSIS MODE. Your goal: map the code to refactor and ALL its dependencies.

DO NOT write or edit any files. Only read, search, and run commands.
You MUST use add_note after each significant finding.

## EXAMPLES OF GOOD ANALYSIS

Example 1 — Extracting a function:
  1. execute_command: "pytest tests/ --tb=no -q" → "48 passed in 2.1s"
  2. add_note: "BASELINE: 48 tests all passing. Must stay at 48 after refactor"
  3. read_file: "src/processor.py" → find the monolithic function
  4. add_note: "processor.py:process() is 130 lines. Three distinct blocks: CSV parsing (lines 20-55), aggregation (lines 56-95), sorting/output (lines 96-130)"
  5. code_search: "from.*processor.*import|import.*processor"
  6. add_note: "CALLERS: main.py:12 imports process(), test_processor.py imports process(). Public API is just process()"
  7. add_note: "SAFE TO EXTRACT: internal helpers _parse_csv(), _aggregate(), _sort_and_format(). The public process() calls them"
  8. phase_complete: "Ready. 3 helpers to extract, 1 public function stays, 48 tests must keep passing"

Example 2 — Renaming/moving:
  1. code_search: "class UserService"
  2. add_note: "UserService in services/user.py. Used by: routes/auth.py, routes/admin.py, tests/test_user.py"
  3. code_search: "UserService" → all references
  4. add_note: "12 references total across 4 files. Need to update all imports after move"
  5. execute_command: "pytest -q" → baseline
  6. add_note: "BASELINE: 92 tests passing"
  7. phase_complete: "UserService used in 4 files, 12 references to update"

## EXAMPLES OF BAD ANALYSIS (DO NOT DO THIS)

Bad — No baseline:
  1. read_file: "src/processor.py"
  2. phase_complete: "Ready to refactor"
  WHY BAD: Didn't run tests first. Won't know if refactor breaks something.

Bad — Missing callers:
  1. read_file: "src/processor.py"
  2. add_note: "Function is 130 lines, should be split"
  3. phase_complete: "Will split the function"
  WHY BAD: Didn't check who calls it. Might break imports or the public API.
"""


_REFACTOR_PLAN = """\
Create a plan for the refactoring. Each step is ONE atomic change. Tests MUST pass after EVERY step.

## RULES
- Never change behavior and structure in the same step
- Include "run full test suite" after EVERY step (not just related tests)
- If renaming: one step to add the new name, one step to update callers, one step to remove old name
- If extracting: one step per extracted function

## EXAMPLES OF GOOD PLANS

Example — Extracting helpers:
[
  {{"step": 1, "description": "Extract _parse_csv() from process() lines 20-55 in processor.py — move code to new function, call it from process()", "files": ["src/processor.py"]}},
  {{"step": 2, "description": "Run pytest tests/ -v to verify no regressions", "files": []}},
  {{"step": 3, "description": "Extract _aggregate() from process() lines 56-95 in processor.py", "files": ["src/processor.py"]}},
  {{"step": 4, "description": "Run pytest tests/ -v to verify no regressions", "files": []}},
  {{"step": 5, "description": "Extract _sort_and_format() from process() lines 96-130 in processor.py", "files": ["src/processor.py"]}},
  {{"step": 6, "description": "Run full test suite to verify all 48 tests still pass", "files": []}}
]

## EXAMPLES OF BAD PLANS (DO NOT DO THIS)

Bad:
[
  {{"step": 1, "description": "Refactor processor.py", "files": ["src/processor.py"]}},
  {{"step": 2, "description": "Run tests", "files": []}}
]
WHY BAD: "Refactor" is not a step. Split into atomic moves with tests between each.
"""


_REFACTOR_EXECUTE = """\
STEP {{step_num}}/{{total_steps}}: {{step_description}}
Files you may modify: {{step_files}}

## RULES
- Make ONE structural change per step
- Use edit_file — do NOT rewrite the entire file
- After editing, ALWAYS run the full test suite (not just one test)
- If any test breaks: undo your change and rethink the approach
- Call step_complete with what changed and test results

## EXAMPLES OF GOOD EXECUTION

Example — Extracting a function:
  1. read_file: "src/processor.py" (lines 20-55 to see exact code to extract)
  2. edit_file: path="src/processor.py",
     old_string="    # Parse CSV\\n    reader = csv.DictReader(...)\\n    rows = []\\n    for row in reader:\\n        ...",
     new_string="    rows = _parse_csv(raw_csv, min_amount, region_filter, include_tax, tax_rate)"
  3. edit_file: path="src/processor.py",
     old_string="def process(raw_csv",
     new_string="def _parse_csv(raw_csv, min_amount, region_filter, include_tax, tax_rate):\\n    reader = csv.DictReader(...)\\n    rows = []\\n    ...\\n    return rows\\n\\n\\ndef process(raw_csv"
  4. execute_command: "pytest tests/ -q"
     → "48 passed"
  5. step_complete: "Extracted _parse_csv(). All 48 tests still pass."

## EXAMPLES OF BAD EXECUTION (DO NOT DO THIS)

Bad:
  write_file: path="src/processor.py", content="(completely rewritten 150-line file)"
  WHY BAD: Rewrote everything instead of making one surgical extract. High risk of regression.
"""


# ═══════════════════════════════════════════════════════════════════════════
# GENERAL / OTHER / SYSADMIN
# ═══════════════════════════════════════════════════════════════════════════

_OTHER_ANALYZE = """\
You are in ANALYSIS MODE. Understand the current state before making changes.

DO NOT write or edit any files. Only read, search, and run commands.
You MUST use add_note after each significant finding.

## EXAMPLES OF GOOD ANALYSIS

Example 1 — Config change:
  1. read_file: "config/settings.yaml"
  2. add_note: "Current timeout: 30s at line 15. Need to change to 60s"
  3. execute_command: "grep -r 'timeout' config/" → check for other timeout refs
  4. add_note: "Only one timeout setting in settings.yaml. No env var override"
  5. phase_complete: "Simple config change at config/settings.yaml:15"

Example 2 — Understanding code:
  1. glob: "src/**/*.py" → list all source files
  2. add_note: "Project structure: src/auth.py, src/models.py, src/routes/, src/utils/"
  3. read_file: "src/routes/api.py"
  4. add_note: "API has 5 routes: /login, /register, /users, /tasks, /health"
  5. code_search: "def authenticate"
  6. add_note: "Auth flow: routes call authenticate() from auth.py which checks JWT"
  7. phase_complete: "Project mapped: FastAPI app with JWT auth, 5 routes"

## EXAMPLES OF BAD ANALYSIS (DO NOT DO THIS)

Bad — No notes:
  1. list_directory: "."
  2. phase_complete: "I see the project"
  WHY BAD: No specific findings noted. Useless for planning.
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
  {{"step": 2, "description": "Verify: execute_command 'grep timeout config/settings.yaml' to confirm change", "files": []}}
]
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
        analyze_prompt=_BUG_ANALYZE,
        plan_prompt=_BUG_PLAN,
        execute_prompt=_BUG_EXECUTE,
        analyze_max_tool_calls=15,
        plan_min_steps=2,
        plan_max_step_files=2,
        execute_max_tool_calls_per_step=6,
        auto_test=True,
        anti_rewrite=False,
    ),
    "feature": PhaseStrategy(
        analyze_prompt=_FEATURE_ANALYZE,
        plan_prompt=_FEATURE_PLAN,
        execute_prompt=_FEATURE_EXECUTE,
        analyze_max_tool_calls=20,
        plan_min_steps=4,
        plan_max_step_files=2,
        execute_max_tool_calls_per_step=8,
        auto_test=True,
        anti_rewrite=True,
    ),
    "refactor": PhaseStrategy(
        analyze_prompt=_REFACTOR_ANALYZE,
        plan_prompt=_REFACTOR_PLAN,
        execute_prompt=_REFACTOR_EXECUTE,
        analyze_max_tool_calls=20,
        plan_min_steps=3,
        plan_max_step_files=3,
        execute_max_tool_calls_per_step=8,
        auto_test=True,
        anti_rewrite=True,
    ),
    "other": PhaseStrategy(
        analyze_prompt=_OTHER_ANALYZE,
        plan_prompt=_OTHER_PLAN,
        execute_prompt=_OTHER_EXECUTE,
        analyze_max_tool_calls=10,
        plan_min_steps=1,
        plan_max_step_files=3,
        execute_max_tool_calls_per_step=8,
        auto_test=False,
        anti_rewrite=False,
    ),
    "sysadmin": PhaseStrategy(
        analyze_prompt=_OTHER_ANALYZE,
        plan_prompt=_OTHER_PLAN,
        execute_prompt=_OTHER_EXECUTE,
        analyze_max_tool_calls=10,
        plan_min_steps=1,
        plan_max_step_files=3,
        execute_max_tool_calls_per_step=8,
        auto_test=False,
        anti_rewrite=False,
    ),
}


def get_strategy(task_type: str) -> PhaseStrategy:
    """Get the phase strategy for a task type. Defaults to 'feature' for unknown types."""
    return STRATEGIES.get(task_type, STRATEGIES["feature"])
