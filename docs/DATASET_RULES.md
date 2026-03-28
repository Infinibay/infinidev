# Infinidev Fine-Tuning Dataset Rules

This document defines ALL rules, patterns, and constraints that training examples must follow.
Every example in the dataset must be 100% correct according to these rules.

---

## 1. Core Workflow

Every task follows: **Understand -> Plan -> Execute -> Verify -> Report**

### 1.1 Exploration-First (MANDATORY)

- The first 1-2 steps MUST be read-only: `read_file`, `code_search`, `glob`, `list_directory`, `execute_command` (read-only).
- NEVER call editing tools (`replace_lines`, `create_file`, `edit_symbol`, `add_symbol`, etc.) until ALL relevant files have been read.
- After reading a file you plan to modify, ALWAYS `add_note` the key structure/line numbers.

### 1.2 Incremental Planning

- Start with 2-3 concrete steps. After each step, add 1-2 more based on findings.
- Growth from 2 to 12+ steps is normal.
- BAD: Planning 8 vague steps upfront.
- GOOD: 2-3 specific steps, execute, add more based on findings.
- Each step MUST name specific files, functions, or commands.

### 1.3 Step Granularity

- Each step = 1-8 tool calls. Split if more needed.
- BAD: "Set up authentication" / "Write the code" / "Test everything"
- GOOD: "Read src/auth.py to find verify_token()" / "Add JWT check to handle_request() in api.py"

---

## 2. Tool Usage Rules

### 2.1 help() — Use Before First Edit (MANDATORY)

- Call `help("edit")` BEFORE the first edit in every task.
- Call `help(tool_name)` whenever unsure about any tool's params.
- Available contexts: `file`, `edit`, `code_intel`, `git`, `shell`, `knowledge`, `web`, or any tool name.

### 2.2 Reading Tools

| Tool | When to Use | Params |
|------|------------|--------|
| `read_file(path)` | Read entire file. Returns numbered lines. Auto-indexes for code intel. | `path` (required) |
| `partial_read(path, start_line, end_line)` | Read a specific line range. Both 1-based inclusive. | `path`, `start_line`, `end_line` (all required) |
| `get_symbol_code(name, file_path?)` | Get source of a function/method/class by name. Faster than read_file + search. | `name` (required), `kind?`, `file_path?` |
| `list_symbols(file_path, kind?)` | List all symbols in a file without reading it. | `file_path` (required), `kind?` (class/method/function/variable) |
| `search_symbols(query, kind?, limit?)` | Search symbols by name across project. Partial matching. | `query` (required), `kind?`, `limit?` |
| `find_references(name, ref_kind?, file_path?)` | Find ALL places a symbol is used. | `name` (required), `ref_kind?`, `file_path?` |

**Rules:**
- DO NOT re-read files that are in `<opened-files>`. They are already up-to-date in the prompt.
- After a write/edit tool, the file is automatically refreshed in `<opened-files>`.
- Always read a file before editing it (to see line numbers).

### 2.3 File Creation

| Tool | When to Use | Params |
|------|------------|--------|
| `create_file(path, content)` | Create a NEW file. FAILS if file already exists. | `path`, `content` (both required) |

**Rules:**
- NEVER use `create_file` to overwrite an existing file. It will error.
- To modify existing files, use `replace_lines`, `edit_symbol`, or insert tools.
- Creates parent directories automatically.

### 2.4 Editing Tools — Line-Based

| Tool | When to Use | Params |
|------|------------|--------|
| `replace_lines(file_path, content, start_line, end_line)` | Replace a range of lines. Deterministic — no text matching. | All required. Lines 1-based inclusive. |
| `add_content_after_line(file_path, line_number, content)` | Insert content AFTER a line. | All required. `line_number` 1-based, 0 = beginning. |
| `add_content_before_line(file_path, line_number, content)` | Insert content BEFORE a line. | All required. `line_number` 1-based. |

**Rules:**
- ALWAYS `read_file` first to get line numbers.
- `replace_lines` with empty `content` deletes the line range.
- After editing, the file is automatically refreshed in `<opened-files>`. Do NOT re-read it.
- `content` must include proper newlines (`\n`).

### 2.5 Editing Tools — Symbol-Based (Preferred for Methods/Functions)

| Tool | When to Use | Params |
|------|------------|--------|
| `edit_symbol(symbol, new_code, file_path?)` | Replace an entire method/function by name. No text matching needed. | `symbol` (required, qualified: "Class.method" or "func_name"), `new_code` (required), `file_path?` |
| `add_symbol(file_path, code, class_name?)` | Add a method/function to a file or class. Auto-indents. | `file_path`, `code` (required), `class_name?` |
| `remove_symbol(symbol, file_path?)` | Remove a method/function by name. | `symbol` (required), `file_path?` |

**Rules:**
- Use qualified names: `"ClassName.method_name"` for methods, `"function_name"` for top-level.
- `new_code` must include the `def` line.
- Indentation is auto-adjusted — provide clean code, the tool matches the original indent.
- If ambiguous (multiple symbols with same name), provide `file_path` to disambiguate.
- Prefer `edit_symbol` over `replace_lines` for replacing entire methods.

### 2.6 Refactoring Tools

| Tool | When to Use | Params |
|------|------------|--------|
| `rename_symbol(symbol, new_name, file_path?)` | Rename definition + ALL references + imports across entire project. | `symbol`, `new_name` (required), `file_path?` |
| `move_symbol(symbol, target_file, target_class?, after_line?)` | Move symbol to another file/class, update imports. | `symbol`, `target_file` (required), `target_class?`, `after_line?` (0 = end) |

**Rules:**
- `rename_symbol` updates definition, all references, and all imports automatically.
- `new_name` must be a valid Python identifier.
- `move_symbol` with `target_class` re-indents code for class body.
- `after_line=0` (default) = insert at end of file/class.
- Both tools re-index all affected files after changes.

### 2.7 Analysis Tools

| Tool | When to Use | Params |
|------|------------|--------|
| `analyze_code(file_path?, checks?)` | Detect errors without AI: broken imports, undefined symbols, unused imports/definitions. | `file_path?` (empty = project), `checks?` (csv) |

**Rules:**
- Fast — runs SQL queries on indexed data, no re-parsing.
- Available checks: `broken_imports`, `undefined_symbols`, `unused_imports`, `unused_definitions`.
- Severities: error (definite), warning (likely), hint (possible).
- Run after creating a file to catch obvious issues before testing.

### 2.8 Exploration Tools

| Tool | When to Use | Params |
|------|------------|--------|
| `list_directory(path)` | See what files exist at a path. | `path` (default "."), `recursive?`, `pattern?` |
| `glob(pattern, path?)` | Find files by glob pattern. | `pattern` (required), `path?`, `content_pattern?`, `max_results?` |
| `code_search(pattern, path?)` | Search text/regex across files. | `pattern` (required), `path?`, `file_extensions?`, `max_results?` |
| `project_structure(path?, depth?)` | Show directory tree with descriptions. | `path?`, `depth?` |

### 2.9 Shell Tools

| Tool | When to Use | Params |
|------|------------|--------|
| `execute_command(command, timeout?, cwd?)` | Run shell commands (build, test, install, etc). | `command` (str, required), `timeout?`, `cwd?` |

**Rules:**
- `command` MUST be a string, never a list.
- Always include the full command: `"cd /path && python -m pytest test_file.py -v"`
- Use for running tests, builds, installations, verification commands.
- Default timeout is 60s. Increase for long builds.

### 2.10 Git Tools

| Tool | When to Use | Params |
|------|------------|--------|
| `git_status()` | Check working tree. | (no params) |
| `git_diff(branch?, file?, staged?)` | Show changes. | All optional |
| `git_branch(branch_name, create?, base_branch?)` | Create/checkout branches. | `branch_name` (required) |
| `git_commit(message, files?)` | Stage and commit. | `message` (required), `files?` |

**Rules:**
- NEVER commit or push unless explicitly asked by user.
- Always run tests before committing.
- Branch naming: `task-{task_id}-<slug>` (lowercase, letters/digits/hyphens).
- Never commit directly to main.

### 2.11 Meta/Pseudo Tools

| Tool | When to Use | Params |
|------|------------|--------|
| `help(context?)` | Get detailed help for any tool or category. | `context?` (None = overview) |
| `step_complete(summary, status, next_steps?, final_answer?)` | MANDATORY after every step. | `summary`, `status` (required) |
| `add_note(note)` | Save key info for later steps. Persists across all steps. | `note` (required, 1-2 sentences) |
| `think(reasoning)` | Reason before acting AND inform the user what you're doing. Free — doesn't count as tool call. | `reasoning` (required) |
| `send_message(message)` | Send progress update to user without ending task. | `message` (required) |

---

## 3. step_complete Rules (CRITICAL)

### 3.1 MUST call after every step. Never skip.

### 3.2 Status Values

| Status | Meaning | Requirements |
|--------|---------|-------------|
| `continue` | More work to do | Must have at least one pending step |
| `done` | Task fully complete | Must provide `final_answer` |
| `blocked` | Cannot proceed | Explain why in summary |
| `explore` | Need to investigate a sub-problem | |

### 3.3 Summary Format (~150 tokens)
```
Read: files read + key findings
Changed: files modified + what changed
Remaining: what still needs doing
Decisions: key choices made and why
```
Skip empty categories. Raw tool output is discarded — only summary survives.

### 3.4 next_steps Operations
```json
{"op": "add", "index": 3, "description": "Run test suite to verify changes"}
{"op": "modify", "index": 2, "description": "Updated: also fix the import in utils.py"}
{"op": "remove", "index": 4}
```
- Only operate on pending steps (not done/skipped).
- Never create speculative steps for uninvestigated work.

### 3.5 final_answer (when status="done")
- Must be complete and helpful — this is what the user sees.
- Summary is internal (user never sees raw output).
- NEVER status="done" without a substantive final_answer.

---

## 4. add_note Rules (CRITICAL)

Context is rebuilt each step. Summaries are ~150 tokens (can't capture everything).

### MUST add_note for:
- File paths and function names discovered
- Key values, error messages, patterns found
- Decisions made and why (avoid reconsideration later)
- Exact line numbers of code planned to edit
- Test results (pass/fail counts)

### Format:
- 1-2 sentences per note
- Max 20 notes per task
- Notes persist across ALL steps, visible in `<notes>` block

---

## 5. think() Rules (IMPORTANT)

The `think` tool serves TWO purposes:
1. **Reasoning** — organize your thoughts before acting.
2. **User communication** — the user SEES your thinking. It's your way of keeping them informed.

### 5.1 When to Use think()
- Before the first edit in a step — explain WHAT you're about to do and WHY.
- After reading a file — summarize what you found and your plan.
- After a test failure — analyze the error before attempting a fix.
- When choosing between approaches — explain trade-offs.
- Before a complex multi-tool sequence — outline the steps.

### 5.2 How to Write think() Content
- Write as if talking to the user: "I can see the bug is in line 45 — the null check is missing. I'll fix it with replace_lines."
- Be concise but informative. 1-3 sentences typical.
- Include the KEY information: what file, what line, what the problem is, what you'll do.

### 5.3 think() Examples

**GOOD — informative, action-oriented:**
```
"The test_create_table test expects auto-increment IDs starting at 1. Looking at _insert_row(), the ID is assigned at line 384 but not included in the row tuple. I'll fix this by moving the ID assignment before the tuple construction."
```

**GOOD — after reading a file:**
```
"read_file shows auth.py has 3 classes: AuthService (line 15), TokenValidator (line 89), and SessionManager (line 145). The bug is likely in TokenValidator.verify() at line 102 — the expiry check uses < instead of <=."
```

**GOOD — after test failure:**
```
"4 tests still failing — all in TestJoins. The LEFT JOIN implementation at line 280 returns empty rows instead of NULL-padded rows. I need to change the else branch to pad with None values."
```

**BAD — empty reasoning:**
```
"Let me fix this."
```

**BAD — too verbose, restating what's obvious:**
```
"I am now going to use the read_file tool to read the file at src/auth.py. This tool will return the contents of the file with line numbers. I will then analyze the contents to find the bug."
```

### 5.4 think() Does NOT Count as a Tool Call
- Use it freely — it never counts toward the step's tool call limit.
- Prefer `think` over wasting a `read_file` call just to "see what's there" when the file is already in `<opened-files>`.

---

## 6. Opened Files Cache Rules

### 5.1 How It Works
- When you `read_file` or edit a file, it's cached in `<opened-files>` in the prompt.
- Written/edited files are **pinned** — they stay for the entire task.
- Read-only files expire after ~20 tool calls.
- After editing, the file is **automatically refreshed** from disk.

### 5.2 MUST Follow
- **DO NOT re-read files that are in `<opened-files>`** — they are already current.
- The content shown IS the current file content, including any edits you made.
- Save tool calls by referencing `<opened-files>` content directly.

---

## 7. Correct Tool Selection by Scenario

### Creating a new file
```
create_file(path="src/validators.py", content="class Validator:\n    ...\n")
```

### Replacing a method entirely
```
read_file(path="src/auth.py")  # get line numbers
edit_symbol(symbol="AuthService.verify_token", new_code="    def verify_token(self, token):\n        ...")
```

### Fixing a single line
```
read_file(path="src/handler.py")  # see line numbers
replace_lines(file_path="src/handler.py", content="    return response.json()\n", start_line=42, end_line=42)
```

### Adding an import at the top
```
read_file(path="src/main.py")  # see current imports
add_content_after_line(file_path="src/main.py", line_number=2, content="import os\n")
```

### Adding a method to a class
```
add_symbol(file_path="src/models.py", code="def validate(self):\n    return bool(self.name)\n", class_name="User")
```

### Deleting lines
```
replace_lines(file_path="src/old.py", content="", start_line=20, end_line=35)
```

### Renaming across project
```
rename_symbol(symbol="old_function", new_name="new_function")
```

### Moving a function to another file
```
move_symbol(symbol="helper_func", target_file="src/utils.py")
```

### Inserting a block before a function
```
read_file(path="src/handler.py")  # get the line where the function starts
add_content_before_line(file_path="src/handler.py", line_number=45, content="# Validation helper\ndef validate_input(data):\n    ...\n\n")
```

---

## 8. Anti-Patterns (NEVER DO THESE)

### 7.1 Tool Misuse

| Wrong | Correct |
|-------|---------|
| `create_file` on existing file | `replace_lines` or `edit_symbol` |
| `read_file` on file already in `<opened-files>` | Reference opened-files content directly |
| `replace_lines` without reading first | Always `read_file` first for line numbers |
| `execute_command(command=["pytest", "-v"])` | `execute_command(command="pytest -v")` — must be string |
| `edit_symbol` with wrong indentation | Provide clean code — tool auto-adjusts indent |
| `step_complete(status="done")` without `final_answer` | Always include substantive final_answer |
| Using `write_file` for existing files | Use `create_file` for new, `replace_lines`/`edit_symbol` for existing |

### 7.2 Workflow Anti-Patterns

| Wrong | Correct |
|-------|---------|
| Editing before reading all relevant files | Explore first, then edit |
| Reading same file 3 times in one step | Read once, use `add_note` to remember key info |
| Planning 8 steps upfront without investigation | Start with 2-3, add more after each step |
| Vague steps ("Fix the code") | Specific: "Fix null check in verify_token() at auth.py:45" |
| No verification after editing | Always run tests or import check after changes |
| Fixing things outside current step | Stay in scope, add new steps for extra work |
| Not using `add_note` for key findings | Always note paths, line numbers, decisions |
| Ignoring context budget warnings | Wrap up when budget > 70% |
| Not searching knowledge base before exploring | `search_findings` first |
| Using `think` after every tool call | Use `think` only when genuinely need to reason |

### 7.3 Code Anti-Patterns

| Wrong | Correct |
|-------|---------|
| Rewriting entire file to change one function | Use `edit_symbol` for the function |
| Adding features not requested | Implement ONLY what was asked |
| Not running tests after changes | Always verify |
| Committing without tests passing | Run tests before commit |
| Force-pushing without explicit instruction | Never use force=true unless told |

---

## 9. Task Type Workflows

### 8.1 Bug Fix

1. `help("edit")` — learn tools
2. `read_file` / `code_search` — find the bug
3. `find_references` — find ALL affected locations
4. `add_note` — record findings
5. `step_complete(status="continue")` — plan the fix
6. `edit_symbol` or `replace_lines` — fix the bug
7. `execute_command("pytest ...")` — verify
8. `step_complete(status="done", final_answer="...")` — report

### 8.2 Feature Implementation

1. `help("edit")` — learn tools
2. `read_file` / `list_directory` / `glob` — understand project structure
3. `add_note` — record key paths and patterns
4. `step_complete(status="continue")` — plan first batch of steps
5. `create_file` — create new files
6. `edit_symbol` / `add_symbol` — modify existing code
7. `execute_command("pytest ...")` — test
8. Iterate: add more steps, implement, test
9. `step_complete(status="done", final_answer="...")` — report

### 8.3 Refactoring

1. `help("edit")` — learn tools
2. `read_file` — understand current code
3. `find_references` — find ALL callers (CRITICAL)
4. `execute_command("pytest ...")` — baseline test count
5. `add_note` — record baseline + all affected files
6. ONE structural change per step:
   - `rename_symbol` for renames
   - `move_symbol` for moves
   - `edit_symbol` + `add_symbol` for extractions
7. `execute_command("pytest ...")` — verify after EVERY change
8. Test count must NEVER decrease

### 8.4 Documentation / Config / Other

1. `read_file` — read current state
2. `replace_lines` or `add_content_after_line` — make changes
3. Verify the change took effect
4. `step_complete(status="done")` — report

---

## 10. JSON Tool Call Format

All tool calls must be valid JSON. Common mistakes to avoid:

### Correct
```json
{"name": "replace_lines", "arguments": {"file_path": "src/main.py", "content": "    return True\n", "start_line": 10, "end_line": 10}}
```

### Wrong — unescaped quotes in content
```json
{"name": "create_file", "arguments": {"path": "test.py", "content": "print("hello")"}}
```
Must be:
```json
{"name": "create_file", "arguments": {"path": "test.py", "content": "print(\"hello\")"}}
```

### Wrong — newlines not escaped
```json
{"name": "create_file", "arguments": {"path": "test.py", "content": "line one
line two"}}
```
Must be:
```json
{"name": "create_file", "arguments": {"path": "test.py", "content": "line one\nline two"}}
```

### Wrong — list instead of string for command
```json
{"name": "execute_command", "arguments": {"command": ["pytest", "-v"]}}
```
Must be:
```json
{"name": "execute_command", "arguments": {"command": "pytest -v"}}
```

---

## 11. Context Budget Awareness

Each iteration has a `<context-budget>` block indicating token usage.

| Budget | Action |
|--------|--------|
| < 70% | Work normally |
| 70-85% | Finish current step. Call `step_complete(status="done")`. Summarize what was done + list remaining work in `final_answer`. |
| > 85% | STOP all tool calls immediately. Call `step_complete(status="done")` with what was completed + what remains. |

**Never ignore context budget. Exceeding the window loses ALL progress.**

---

## 12. Knowledge Base Usage

### When to Record (`record_finding`)
- Project structure discoveries (key dirs, entry points, patterns)
- Bug root causes and tricky behaviors
- Research results (API details, solutions found)

### When to Search (`search_findings`)
- BEFORE exploring code — check if you already know about it
- BEFORE researching online — check if answer already found
- When user asks about something — check prior context

### Finding Types
- `project_context`: Structure, patterns, conventions (confidence 0.8-1.0)
- `observation`: Bug findings, tricky behaviors
- `conclusion`: Research results, solutions

---

## 13. Security Rules

- Sanitize external input — never trust user input without validation
- Never concatenate user values into SQL, shell commands, or prompts
- Use parameterized queries, `shlex.quote`, subprocess with lists
- Avoid `pickle`, `yaml.load` (use `safe_load`), `eval`, `exec` on untrusted data
- Never log/print secrets, tokens, API keys, passwords
- Validate file paths to prevent directory traversal

---

## 14. Dataset Example Structure

Each training example should follow this format:

```
System prompt: [CLI_AGENT_IDENTITY + tool schemas]
User prompt: [task description + <opened-files> + <notes> + <plan> + <previous-actions>]
Assistant response: [tool calls in correct order]
```

### Minimum Tool Calls Per Example Type

| Scenario | Min Tools | Must Include |
|----------|----------|-------------|
| Read + understand | 2-4 | read_file, add_note, step_complete |
| Bug fix (single file) | 4-8 | help("edit"), read_file, edit_symbol/replace_lines, execute_command, step_complete |
| Feature (new file) | 5-10 | help("edit"), read_file, create_file, execute_command, step_complete |
| Feature (modify existing) | 5-10 | help("edit"), read_file, edit_symbol/add_symbol, execute_command, step_complete |
| Refactoring | 6-12 | help("edit"), read_file, find_references, edit_symbol/rename_symbol, execute_command, step_complete |
| Investigation (read-only) | 3-6 | read_file/code_search, add_note, step_complete |

### Every Example MUST:
1. Start with `help("edit")` on the first editing step
2. Read files before editing them
3. NOT re-read files in `<opened-files>`
4. Use correct tool for the scenario (see Section 6)
5. End with `step_complete`
6. Have valid JSON in all tool call arguments
7. Include `\n` in content strings (never raw newlines in JSON)
8. Use string for `execute_command.command` (never list)
9. Use `add_note` for any finding that will be needed in later steps
10. Run tests after code changes
