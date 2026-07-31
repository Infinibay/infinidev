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
    "stop_planning_start_coding": GuidanceEntry(
        key="stop_planning_start_coding",
        title="Stop planning, start opening files",
        body=(
            "You have created multiple plan steps but you have not "
            "written or modified a single file yet. Planning more is "
            "procrastination. Pick the file your active step names, "
            "call read_file or list_symbols on it, then make the actual "
            "change with edit_file or create_file. "
            "Tests, refactors, and features all require BYTES landing "
            "on disk — a perfect plan with zero edits ships nothing. "
            "If you genuinely don't know what file to touch yet, your "
            "active step is too vague — rewrite it with a concrete file "
            "path and call read_file on that path right now."
        ),
        example=(
            "BAD pattern (what you are doing now):\n"
            "  add_step(title='Plan refactoring approach')\n"
            "  add_step(title='Identify cohesive groups')\n"
            "  add_step(title='Design new module structure')\n"
            "  add_step(title='Document the changes')\n"
            "  ... 0 edits to any file ...\n\n"
            "GOOD pattern:\n"
            "  read_file('app/services/Foo.ts')   # see the actual code\n"
            "  edit_file(file_path='app/services/Foo.ts',\n"
            "            old_string='...', new_string='...')"
        ),
    ),
    "stuck_on_planning": GuidanceEntry(
        key="stuck_on_planning",
        title="How to write a concrete plan step",
        body=(
            "Each step you create with add_step must name a FILE PATH, a "
            "FUNCTION (with parens), and the SPECIFIC change. Vague titles "
            "like 'implement the feature' get flagged as warnings, and a "
            "vague step is where a run drifts. Set expected_output "
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
    "stop_reading": GuidanceEntry(
        key="stop_reading",
        title="Stop exploring — start editing",
        body=(
            "Your fingerprint for this step shows lots of reads, "
            "lookups, or get_symbol_code calls and ZERO edits. The "
            "model that wins this kind of task is the one that "
            "commits to a fix early and iterates on it, not the one "
            "that keeps gathering more context.\n\n"
            "What to do RIGHT NOW:\n"
            "  1. Re-read your own notes / discovered_context. You "
            "already know enough.\n"
            "  2. Pick the SMALLEST change that addresses the step.\n"
            "  3. Make it with edit_file / create_file. Even a "
            "     wrong first attempt is more "
            "     useful than another read — the test failure that "
            "     follows will tell you what's actually wrong.\n"
            "  4. Run the test covering the file you just edited (or a "
            "     static check on it) and let "
            "     the result drive the next iteration.\n\n"
            "NEVER do these:\n"
            "  - read_file on a file you've already opened in this task\n"
            "  - get_symbol_code on a symbol whose source is in <opened-files>\n"
            "  - search_symbols / find_references for terms you've already searched\n"
            "  - add_step to break the work down further — your plan is fine, EXECUTE it"
        ),
        example=(
            "If your last 8 tool calls were:\n"
            "  read_file, get_symbol_code, get_symbol_code, "
            "read_file, code_search, get_symbol_code, list_symbols, "
            "read_file\n\n"
            "...your next call MUST be one of:\n"
            "  edit_file / create_file\n"
            "or step_complete with status='blocked' if you genuinely cannot proceed."
        ),
    ),
    "python_env_mismatch": GuidanceEntry(
        key="python_env_mismatch",
        title="ImportError from python/pytest means a virtualenv mismatch",
        body=(
            "When `python -c \"import X\"` or `pytest ...` raises "
            "ImportError / ModuleNotFoundError immediately after a "
            "successful `pip install`, the install and the import are "
            "hitting different Python environments. "
            "Reinstalling deps will NOT help — the package IS installed, "
            "just in the wrong venv. STOP installing things and check "
            "the environment first.\n\n"
            "Run `which python && python --version && python -c "
            "\"import sys; print(sys.executable, sys.path[:3])\"` to see "
            "exactly which interpreter is running. If it points at the "
            "agent's host venv, look for the project's own venv at "
            "`<repo>/.venv/bin/python` or `<repo>/venv/bin/python` and use "
            "that absolute path. Read setup.cfg, pyproject.toml or tox.ini "
            "for how this project runs its tests before guessing.\n\n"
            "IF no project venv exists, THEN work without running tests: "
            "read the code, make the fix, and read the failing test file "
            "to confirm your change matches what it asserts. NEVER spend "
            "the step budget debugging the host environment."
        ),
        example=(
            "1. execute_command('which python && python -c \"import sys; "
            "print(sys.executable)\"')\n"
            "   Output: /home/andres/infinidev/.venv/bin/python\n"
            "2. execute_command('ls -d ./.venv ./venv 2>/dev/null')\n"
            "   Output: ./venv\n"
            "3. execute_command('./venv/bin/python -c \"import flask; "
            "print(flask.__version__)\"')\n"
            "   Output: 2.0.1   (correct env)\n"
            "4. From here on, prefix all python invocations with "
            "./venv/bin/python instead of bare `python`."
        ),
    ),
    "stuck_on_edit": GuidanceEntry(
        key="stuck_on_edit",
        title="How to edit existing files in this project",
        body=(
            "Use edit_file: it takes the exact text to replace "
            "(old_string) and what to put there (new_string). Read the "
            "file in this step and copy old_string from what you read — "
            "it must match byte for byte, indentation included, and it "
            "must appear exactly once. If the edit is refused for being "
            "ambiguous, add the lines above and below until the match is "
            "unique; if it is refused for not matching, read the file "
            "again rather than guessing at the text. The pre-write "
            "syntax check rejects an edit that leaves invalid Python — "
            "fix the indentation and retry, do not work around it. For "
            "brand-new files use create_file."
        ),
        example=(
            "1. read_file('src/auth.py')   # see verify_token\n"
            "2. edit_file(\n"
            "     file_path='src/auth.py',\n"
            "     old_string='    return payload[\"exp\"]',\n"
            "     new_string='    return payload[\"exp\"] is not None',\n"
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
            "open the file named in that failure, THEN edit. Patching blindly without "
            "reading the failure is the main reason a small model loops "
            "on the same broken edit."
        ),
        example=(
            "1. execute_command('pytest tests/test_x.py::test_foo -v')\n"
            "2. tail_test_output(mode='structured')\n"
            "   Output: {failures: [{test_name: '...test_foo', file: 'handler.py',\n"
            "                  line: 52, error_type: 'AssertionError',\n"
            "                  message: 'expected 200, got 404'}]}\n"
            "3. add_note('test_foo: handler.py:52 returns 404 not 200 — route missing')\n"
            "4. read_file('src/handler.py')\n"
            "5. edit_file(file_path='src/handler.py', old_string=..., new_string=...)\n"
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
            "   Output: failures=[{file: 'minidb.py', line: 92,\n"
            "                error_type: 'TypeError',\n"
            "                message: '_parse_values: quote_char is None'}]\n"
            "3. add_note('TypeError at minidb.py:92 inside _parse_values')\n"
            "4. read_file('minidb.py')   # focus on _parse_values around line 92\n"
            "5. modify_step(index=N, expected_output='_parse_values handles empty values')\n"
            "6. edit_file('minidb.py', old_string=..., new_string=...)"
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
            "  edit_file(file_path='src/auth.py', old_string=..., new_string=...)"
        ),
    ),
    "unknown_tool": GuidanceEntry(
        key="unknown_tool",
        title="You are calling tools that don't exist",
        body=(
            "The error 'Unknown tool: X' means X is not in the registered "
            "toolset for this run. Do NOT keep retrying it. Call the help "
            "tool to see what is available, or pick from these common "
            "ones: read_file, edit_file, create_file, "
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
            "a file, run a command) and emit it as a tool call. IF you do "
            "not know which one, THEN call read_file on the file named in "
            "your current step, or list_directory on the workspace."
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
            "code_search(query='auth', path='src/') returns 5 hits\n"
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
            "revert that hunk (edit_file it back to the previous content) "
            "and re-think the change more carefully.\n"
            "  4. If the new failure is in a DIFFERENT file, the edit "
            "had a side effect — read the new failing file and trace why."
        ),
        example=(
            "before edit: pytest test_minidb.py gave 1 failed, 1 passed\n"
            "after  edit: pytest test_minidb.py gave 2 failed, 0 passed   REGRESSION\n"
            "\n"
            "1. tail_test_output(mode='structured')\n"
            "   Newly failing: TestCreateTable.test_create_simple at minidb.py:84\n"
            "2. read_file('minidb.py')   # focus around line 84\n"
            "3. The line you just changed is the problem — revert it.\n"
            "4. edit_file('minidb.py', old_string=<what you just wrote>,\n"
            "              new_string=<the previous working version>)\n"
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
            "   Output: {failure_count: 1, failures: [{file: 'src/x.py',\n"
            "        line: 42, error_type: 'KeyError', message: \"'id'\"}]}\n"
            "3. add_note('test fails: KeyError id at src/x.py:42')\n"
            "4. read_file('src/x.py')   # focus around line 42\n"
            "5. edit_file(file_path='src/x.py', old_string=..., new_string=...)\n"
            "6. execute_command('pytest tests/test_x.py -v')   # re-verify"
        ),
    ),
    "duplicate_steps": GuidanceEntry(
        key="duplicate_steps",
        title="Your plan has near-duplicate steps — clean it up",
        body=(
            "Several steps in your plan have nearly identical titles "
            "(e.g. 'Read test files to understand behavior' and 'Read "
            "test_minidb.py to understand required cases'). That "
            "means you re-planned the same work without removing the "
            "previous steps. The plan does not get smarter by accumulating "
            "drafts — it gets noisy and the model loses track of where it "
            "is. Use remove_step on the duplicates and modify_step to "
            "differentiate the ones that remain. Each step describes "
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
