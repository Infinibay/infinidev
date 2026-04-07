"""Pre-baked guidance entries for the stuck-pattern detectors.

This module is intentionally pure data — no imports from the rest of
the engine, no detection logic, no I/O. Adding a new entry is one
literal in :data:`_LIBRARY`. Each entry is a short title + body +
optional concrete example, capped at ~250 tokens so the cumulative
context overhead stays small even at the per-task max (3 entries).

The keys MUST match the keys returned by the detectors in
``engine.guidance.detectors`` — that's the only contract between the
two modules.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuidanceEntry:
    """A single piece of pre-baked how-to advice for a stuck-pattern."""

    key: str
    title: str
    body: str
    example: str = ""

    def render(self) -> str:
        """Format the entry as a self-contained ``<guidance>`` XML block."""
        parts = [f"<guidance pattern=\"{self.key}\">", f"## {self.title}", "", self.body]
        if self.example:
            parts.extend(["", "Concrete example:", "```", self.example, "```"])
        parts.append("</guidance>")
        return "\n".join(parts)


_LIBRARY: dict[str, GuidanceEntry] = {
    "stuck_on_planning": GuidanceEntry(
        key="stuck_on_planning",
        title="How to write a concrete plan step",
        body=(
            "Each step you create with add_step must name a FILE PATH, a "
            "FUNCTION (with parens), and the SPECIFIC change. Vague titles "
            "like 'implement the feature' get flagged as warnings and the "
            "model that wrote them tends to drift. Set expected_output "
            "to a short, verifiable check the step must pass."
        ),
        example=(
            "BAD:  add_step(title='Implement the auth module')\n"
            "GOOD: add_step(\n"
            "        title='Add JWT exp check to verify_token() in auth.py:52',\n"
            "        expected_output='pytest tests/test_auth.py::test_expired passes',\n"
            "      )"
        ),
    ),
    "stuck_on_edit": GuidanceEntry(
        key="stuck_on_edit",
        title="How to edit existing files in this project",
        body=(
            "Use replace_lines for surgical edits — it takes a file_path "
            "plus start_line, end_line, and the replacement content. "
            "Read the file first to find the exact line numbers; do not "
            "guess. The pre-write syntax check will reject the edit if "
            "the result has invalid Python — fix the indentation and "
            "retry, do not work around it. For brand-new files use "
            "create_file. For renaming a function, use edit_symbol."
        ),
        example=(
            "1. read_file('src/auth.py')   # see verify_token at line 42-58\n"
            "2. replace_lines(\n"
            "     file_path='src/auth.py',\n"
            "     start_line=52, end_line=52,\n"
            "     content='    return payload[\"exp\"] is not None\\n',\n"
            "   )\n"
            "3. execute_command('pytest tests/test_auth.py -v')"
        ),
    ),
    "stuck_on_tests": GuidanceEntry(
        key="stuck_on_tests",
        title="When tests keep failing, READ the failure output",
        body=(
            "exit_code tells you nothing useful — the actual error message "
            "is in stdout/stderr. The fastest way to read it is "
            "tail_test_output(mode='structured'), which returns the parsed "
            "failures from the last test run as a JSON list with "
            "test_name + file + line + error_type + message. No shell "
            "pipes, no scrolling. Works for pytest, jest/vitest, mocha, "
            "cargo test, go test, rspec, and node:test.\n\n"
            "After reading the structured failures, add_note the EXACT "
            "failure mode (file:line + what was expected vs actual), THEN "
            "open the relevant file, THEN edit. Patching blindly without "
            "reading the failure is the main reason a small model loops "
            "on the same broken edit."
        ),
        example=(
            "1. execute_command('pytest tests/test_x.py::test_foo -v')\n"
            "2. tail_test_output(mode='structured')\n"
            "   → {failures: [{test_name: '...test_foo', file: 'handler.py',\n"
            "                  line: 52, error_type: 'AssertionError',\n"
            "                  message: 'expected 200, got 404'}]}\n"
            "3. add_note('test_foo: handler.py:52 returns 404 not 200 — route missing')\n"
            "4. read_file('src/handler.py')\n"
            "5. replace_lines(file_path='src/handler.py', start_line=52, ...)\n"
            "6. execute_command('pytest tests/test_x.py::test_foo -v')"
        ),
    ),
    "same_test_output_loop": GuidanceEntry(
        key="same_test_output_loop",
        title="Your edits are not changing the test outcome — switch tactics",
        body=(
            "You have run the test runner 3+ times and the pass/fail count "
            "is IDENTICAL each time. Your edits are not affecting the "
            "failing test. This means EITHER (a) you are editing the wrong "
            "file or wrong line, OR (b) the bug is somewhere you haven't "
            "looked yet.\n\n"
            "STOP editing and do a diagnostic step:\n"
            "1) isolate ONE failing test (e.g. `pytest path::name -v`, "
            "`jest -t 'name'`, `cargo test name`, `go test -run TestName`),\n"
            "2) call tail_test_output(mode='structured') to get the parsed "
            "failure with the EXACT file:line where the error is raised — "
            "this is the canonical way to read failures across pytest/"
            "jest/cargo/go/rspec/node:test, no shell pipes needed,\n"
            "3) add_note the file:line and what was expected vs actual,\n"
            "4) read THAT file at THAT line. Only then edit."
        ),
        example=(
            "1. execute_command('pytest tests/test_foo.py::test_one -v')\n"
            "2. tail_test_output(mode='structured')\n"
            "   → failures=[{file: 'minidb.py', line: 92,\n"
            "                error_type: 'TypeError',\n"
            "                message: '_parse_values: quote_char is None'}]\n"
            "3. add_note('TypeError at minidb.py:92 inside _parse_values')\n"
            "4. read_file('minidb.py')   # focus on _parse_values around line 92\n"
            "5. modify_step(index=N, expected_output='_parse_values handles empty values')\n"
            "6. replace_lines('minidb.py', start_line=88, end_line=95, content=...)"
        ),
    ),
    "reread_loop": GuidanceEntry(
        key="reread_loop",
        title="Stop re-reading the same file",
        body=(
            "You have already read this file recently. Its content is in "
            "<opened-files> in your prompt. Re-reading wastes tokens and "
            "the file has not changed since the last read. Either edit it, "
            "or use add_note to save what you found and move on. If you "
            "need to see only a specific range, use read_file with offset "
            "and limit instead of fetching the whole file again."
        ),
        example=(
            "Wrong: read_file('src/auth.py') × 3\n"
            "Right:\n"
            "  read_file('src/auth.py')\n"
            "  add_note('verify_token at auth.py:42, missing exp check')\n"
            "  replace_lines(file_path='src/auth.py', start_line=52, ...)"
        ),
    ),
    "unknown_tool": GuidanceEntry(
        key="unknown_tool",
        title="You are calling tools that don't exist",
        body=(
            "The error 'Unknown tool: X' means X is not in the registered "
            "toolset for this run. Do NOT keep retrying it. Call the help "
            "tool to see what is available, or pick from these common "
            "ones: read_file, replace_lines, edit_file, create_file, "
            "code_search, glob, list_directory, execute_command, "
            "add_note, add_step, modify_step, step_complete."
        ),
        example=(
            "Wrong: search_in_files(...)        # not a real tool\n"
            "Right: code_search(query='verify_token', path='src/')"
        ),
    ),
    "vague_steps": GuidanceEntry(
        key="vague_steps",
        title="Your steps are too vague to act on",
        body=(
            "Several of your add_step calls did not name a file path, "
            "function, or line — that is why the tool returned a warning "
            "and you cannot make progress. Use modify_step on each vague "
            "step to add a concrete file:line and the specific change. "
            "Then set expected_output to one short verifiable sentence."
        ),
        example=(
            "modify_step(\n"
            "  index=2,\n"
            "  title='Add count_records() to minidb.py:24 returning len(self.tables)',\n"
            "  expected_output='pytest tests/test_count.py::test_basic passes',\n"
            ")"
        ),
    ),
    "text_only_iters": GuidanceEntry(
        key="text_only_iters",
        title="You must call tools, not just write text",
        body=(
            "Your last responses contained text only and no tool calls. "
            "The loop cannot make progress without tool calls. Pick the "
            "single most useful next action right now (read a file, edit "
            "a file, run a command) and emit it as a tool call. If you "
            "are unsure what to do, call read_file on the most relevant "
            "file or list_directory on the workspace."
        ),
        example=(
            "Wrong:\n"
            "  'I should now read the auth module to check the token logic.'\n"
            "Right (actual tool call, not narration):\n"
            "  read_file('src/auth.py')"
        ),
    ),
    "stuck_on_search": GuidanceEntry(
        key="stuck_on_search",
        title="Stop searching, start reading",
        body=(
            "You have run multiple searches without opening any of the "
            "results. The next step is to pick the most promising hit "
            "and read_file on it. If the searches all returned nothing, "
            "the term you are searching for does not exist in this "
            "codebase — try a synonym or use list_directory to discover "
            "the structure first."
        ),
        example=(
            "code_search(query='auth', path='src/') -> 5 hits\n"
            "read_file('src/auth/handlers.py')   # actually read the top hit\n"
            "add_note('auth flow lives in src/auth/handlers.py:42 verify()')"
        ),
    ),
    "malformed_tool_call": GuidanceEntry(
        key="malformed_tool_call",
        title="You wrote a tool call as text — call the tool, don't print it",
        body=(
            "You emitted something like ``{\"tool_calls\": [{\"name\": ...}]}`` "
            "or ``{\"name\": \"read_file\", \"arguments\": {...}}`` inside "
            "your normal text/thinking output. That JSON does NOT call "
            "the tool — it's just a string the engine throws away. To "
            "actually run a tool you have to emit it through the "
            "function-calling channel: stop writing tool-call JSON in "
            "your text, and instead emit the tool call as a real "
            "function call. The engine will then dispatch it.\n\n"
            "If your model template doesn't expose function calling at "
            "all, the engine falls back to manual mode and accepts "
            "exactly one shape per response: a single JSON object "
            "``{\"name\": \"...\", \"arguments\": {...}}`` and NOTHING ELSE "
            "in the content. No prose around it, no markdown fences, "
            "no nested ``tool_calls`` array, no commentary."
        ),
        example=(
            "WRONG (text fragment, ignored):\n"
            "  Let me read the file.\n"
            "  {\"tool_calls\": [{\"name\": \"read_file\",\n"
            "                    \"arguments\": {\"file_path\": \"x.py\"}}]}\n"
            "\n"
            "RIGHT (real function call):\n"
            "  read_file(file_path=\"x.py\")\n"
            "\n"
            "RIGHT (manual fallback — single bare object, no prose):\n"
            "  {\"name\": \"read_file\", \"arguments\": {\"file_path\": \"x.py\"}}"
        ),
    ),
    "regression_after_edit": GuidanceEntry(
        key="regression_after_edit",
        title="Your last edit broke a test that was previously passing",
        body=(
            "You just ran the same test command twice and the second "
            "result has more failures than the first. Whatever you "
            "edited between the two runs broke a test that was working. "
            "This is the most expensive class of mistake — you destroyed "
            "real progress and replaced it with a problem that didn't "
            "exist before.\n\n"
            "Stop editing forward. Recover the previous state of the "
            "file you just modified:\n"
            "  1. Read the file you just edited and identify the lines "
            "you changed in the last edit.\n"
            "  2. Run tail_test_output(mode='structured') to see WHICH "
            "test newly fails and on what file:line.\n"
            "  3. If the new failure points to a line you just touched, "
            "revert that hunk (replace_lines back to the previous content) "
            "and re-think the change more carefully.\n"
            "  4. If the new failure is in a DIFFERENT file, the edit "
            "had a side effect — read the new failing file and trace why."
        ),
        example=(
            "before edit: pytest test_minidb.py → 1 failed, 1 passed\n"
            "after  edit: pytest test_minidb.py → 2 failed, 0 passed   ← regression\n"
            "\n"
            "1. tail_test_output(mode='structured')\n"
            "   → newly failing: TestCreateTable.test_create_simple at minidb.py:84\n"
            "2. read_file('minidb.py')   # focus around line 84\n"
            "3. The line you just changed is the problem — revert it.\n"
            "4. replace_lines('minidb.py', start_line=82, end_line=86,\n"
            "                 content=<the previous working version>)\n"
            "5. execute_command('pytest test_minidb.py')   # confirm 1/2 again\n"
            "6. THEN reattempt the original fix more carefully."
        ),
    ),
    "first_test_run": GuidanceEntry(
        key="first_test_run",
        title="You just ran tests — here's the fastest way to read the result",
        body=(
            "After any execute_command that runs a test runner (pytest, "
            "jest, vitest, mocha, cargo test, go test, rspec, node:test, "
            "etc.), call tail_test_output(mode='structured') to get the "
            "parsed failures as a JSON list with test_name + file + line "
            "+ error_type + message. This is the fastest way to read "
            "what failed across every supported runner — no shell pipes, "
            "no scrolling, no re-running the test. Use it BEFORE deciding "
            "your next edit, so the edit targets the right file:line."
        ),
        example=(
            "1. execute_command('pytest tests/test_x.py -v')\n"
            "2. tail_test_output(mode='structured')\n"
            "   → {failure_count: 1, failures: [{file: 'src/x.py',\n"
            "        line: 42, error_type: 'KeyError', message: \"'id'\"}]}\n"
            "3. add_note('test fails: KeyError id at src/x.py:42')\n"
            "4. read_file('src/x.py')   # focus around line 42\n"
            "5. replace_lines(file_path='src/x.py', start_line=40, ...)\n"
            "6. execute_command('pytest tests/test_x.py -v')   # re-verify"
        ),
    ),
    "duplicate_steps": GuidanceEntry(
        key="duplicate_steps",
        title="Your plan has near-duplicate steps — clean it up",
        body=(
            "Several steps in your plan have nearly identical titles "
            "(e.g. 'Read test files to understand behavior' and 'Read "
            "test_minidb.py to understand required cases'). This usually "
            "means you re-planned the same work without removing the "
            "previous steps. The plan does not get smarter by accumulating "
            "drafts — it gets noisy and the model loses track of where it "
            "is. Use remove_step on the duplicates and modify_step to "
            "differentiate the ones that remain. Each step should describe "
            "a UNIQUE action with its own file:line and expected_output."
        ),
        example=(
            "Plan looks like:\n"
            "  3. Read test files to understand behavior\n"
            "  4. Read test_minidb.py to understand cases\n"
            "  5. Read test files to understand expected behavior\n"
            "Fix it:\n"
            "  remove_step(index=4)\n"
            "  remove_step(index=5)\n"
            "  modify_step(index=3,\n"
            "    title='Read test_minidb.py:69-94 to list TestCreateTable assertions',\n"
            "    expected_output='I can name each test method and what it asserts')"
        ),
    ),
}


def get_entry(key: str) -> GuidanceEntry | None:
    """Return the entry for *key* or None if not in the library."""
    return _LIBRARY.get(key)


__all__ = ["GuidanceEntry", "get_entry"]
