"""Meta-tool that provides detailed help and examples for all tools."""

from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


from infinidev.tools.meta.help_input import HelpInput

# ---------------------------------------------------------------------------
# Help content: categories and individual tools
# ---------------------------------------------------------------------------

_CATEGORY_INDEX = {
    "file": ["read_file", "create_file", "replace_lines", "add_content_after_line", "add_content_before_line", "apply_patch", "list_directory", "glob", "code_search"],
    "code_intel": ["get_symbol_code", "list_symbols", "search_symbols", "find_references", "project_structure", "analyze_code"],
    "edit": ["edit_symbol", "add_symbol", "remove_symbol", "replace_lines", "add_content_after_line", "add_content_before_line", "apply_patch", "rename_symbol", "move_symbol"],
    "git": ["git_branch", "git_commit", "git_diff", "git_status"],
    "shell": ["execute_command", "code_interpreter"],
    "knowledge": ["record_finding", "read_findings", "search_findings", "search_knowledge"],
    "web": ["web_search", "web_fetch"],
}

HELP_CONTENT: dict[str | None, str] = {
    # ── Overview ──────────────────────────────────────────────────────────
    None: """\
Available help categories:
  file        — Reading and creating files
  edit        — Editing existing files (symbolic + line-based)
  code_intel  — Code intelligence: symbols, definitions, references
  git         — Git operations
  shell       — Shell commands and code interpreter
  knowledge   — Findings and knowledge base
  web         — Web search and fetch

Call help(context="<category>") for tool list, or help(context="<tool_name>") for detailed usage.""",

    # ── Category: file ────────────────────────────────────────────────────
    "file": """\
FILE TOOLS — Reading and creating files

  read_file(file_path, start_line?, end_line?)
    Read file with numbered lines. Pass start_line/end_line for a
    specific range (both 1-based, inclusive). For files >800 lines
    without a range, returns a structured skeleton.

  create_file(file_path, content)
    Create a new file. FAILS if the file already exists.

  replace_lines(file_path, content, start_line, end_line)
    Replace a range of lines with new content. See help("replace_lines").

  add_content_after_line(file_path, line_number, content)
    Insert content after a specific line.

  add_content_before_line(file_path, line_number, content)
    Insert content before a specific line.

  apply_patch(patch, strip?)
    Apply a unified diff to one or more files. See help("apply_patch").

  list_directory(file_path?)
    List files and directories. Defaults to workspace root.

  glob(pattern, file_path?)
    Find files matching a glob pattern.

  code_search(pattern, file_path?, file_extensions?)
    Search for text/regex patterns across files.

TIP: Always read_file first to get line numbers, then use replace_lines or edit_symbol to edit.""",

    # ── Category: edit ────────────────────────────────────────────────────
    "edit": """\
EDIT TOOLS — Modifying existing files

TWO APPROACHES:

1. SYMBOLIC (preferred for methods/functions):
   edit_symbol(symbol, new_code)      — Replace a method/function by name
   add_symbol(file_path, code, class_name?) — Add a method to a file or class
   remove_symbol(symbol)              — Remove a method by name

2. LINE-BASED (for anything else):
   replace_lines(file_path, content, start_line, end_line)
   — Deterministic: specify exact line range, no text matching needed
   — Use read_file first to see line numbers

3. INSERT (add new content without removing):
   add_content_after_line(file_path, line_number, content)
   add_content_before_line(file_path, line_number, content)

4. PATCH (multi-file changes as unified diff):
   apply_patch(patch, strip?) — Apply a unified diff to one or more files

5. REFACTORING (project-wide operations):
   rename_symbol(symbol, new_name)  — Rename everywhere: definition + all references + imports
   move_symbol(symbol, target_file) — Move to another file/class, update imports

WORKFLOW:
  1. read_file(file_path="src/foo.py")           → see the code with line numbers
  2. edit_symbol(symbol="Foo.bar", new_code=...)  → replace method by name
  OR
  2. replace_lines(file_path="src/foo.py", content="new code", start_line=10, end_line=25)""",

    # ── Category: code_intel ──────────────────────────────────────────────
    "code_intel": """\
CODE INTELLIGENCE TOOLS — Symbol-based code navigation

  get_symbol_code(name, file_path?)
    Get the full source code of a function, method, or class by name.
    Use qualified names: "ClassName.method_name" or just "function_name".

  list_symbols(file_path, kind?)
    List all symbols in a file. Filter by kind: "class", "method", "function", "variable".

  search_symbols(query, kind?)
    Search for symbols by name across the project. Supports partial matching.

  find_references(name, ref_kind?)
    Find all places where a symbol is used. Essential before renaming or removing.

  project_structure(file_path?, depth?)
    Show project directory tree with file descriptions.

  analyze_code(file_path?, checks?)
    Detect broken imports, undefined symbols, unused code.

TIP: Files are auto-indexed when you read_file them. Symbol tools work best after reading.""",

    # ── Category: git ─────────────────────────────────────────────────────
    "git": """\
GIT TOOLS

  git_status()                              — Show working tree status
  git_diff(branch?, file?, staged?)         — Show changes (unstaged by default)
  git_branch(branch_name, create?, base_branch?) — Create or checkout branches
  git_commit(message, files?)               — Commit changes""",

    # ── Category: shell ───────────────────────────────────────────────────
    "shell": """\
SHELL TOOLS

  execute_command(command, cwd?, timeout?, env?)
    Run a shell command. Use for: running tests, installing packages, build commands.

  code_interpreter(code, libraries_used?, timeout?)
    Execute Python code snippets in an isolated environment.""",

    # ── Category: knowledge ───────────────────────────────────────────────
    "knowledge": """\
KNOWLEDGE TOOLS — Findings and reports

  record_finding(title, content, finding_type?, confidence?, tags?, sources?)
    Save a finding with metadata.

  read_findings(query?, finding_type?, limit?)
    Read stored findings. Filter by query or type.

  search_findings(query, limit?)
    Semantic search over findings.

  search_knowledge(query, sources?, limit?)
    Search across findings + docs.""",

    # ── Category: web ─────────────────────────────────────────────────────
    "web": """\
WEB TOOLS

  web_search(query, num_results?)  — Search the web
  web_fetch(url, format?)          — Fetch content from a URL (markdown or text)""",

    # ── Individual tools ──────────────────────────────────────────────────

    "read_file": """\
read_file(file_path, offset?, limit?)

Read the full contents of a file. Returns numbered lines for easy reference.
Automatically indexes the file for code intelligence (symbol lookup, etc).

PARAMS:
  file_path (str, required) — File path (absolute or relative to workspace)
  offset (int, optional)    — Line number to start reading from (1-based). Default: start of file.
  limit (int, optional)     — Maximum number of lines to read. Default: entire file.

RETURNS: Numbered lines in format "  LINE_NUM\\tCONTENT"

EXAMPLES:
  read_file(file_path="src/auth.py")
  read_file(file_path="src/auth.py", start_line=10, end_line=30)  # specific range
  read_file(file_path="src/big_file.py", offset=100, limit=50)    # lines 100-149

TIPS:
  - For large files, pass start_line/end_line to read only the part you need
  - Files larger than ~800 lines return a structured skeleton instead of
    raw content — use the L<n>-<m> ranges from the skeleton to zoom in
  - Line numbers in the output can be used with replace_lines for editing
  - Binary files are automatically detected and rejected""",

    "create_file": """\
create_file(file_path, content)

Create a new file. FAILS if the file already exists. Creates parent directories as needed.

PARAMS:
  file_path (str, required) — Path for the new file
  content (str, required)   — Content to write

EXAMPLES:
  create_file(file_path="src/utils/helpers.py", content="def greet(name):\\n    return f'Hello, {name}!'\\n")
  create_file(file_path="tests/test_helpers.py", content="import pytest\\n...")

NOTE: To modify existing files, use replace_lines or edit_symbol instead.""",

    "replace_lines": """\
replace_lines(file_path, content, start_line, end_line)

Replace a range of lines with new content. Deterministic — no text matching, no retries.
Always read_file first to see line numbers.

PARAMS:
  file_path (str, required)  — Path to the file
  content (str, required)    — New content (replaces the line range)
  start_line (int, required) — First line to replace (1-based, inclusive)
  end_line (int, required)   — Last line to replace (1-based, inclusive)

SPECIAL CASES:
  - Empty content: deletes the lines (content="")
  - start_line == end_line + 1: inserts without removing lines

EXAMPLES:
  # Replace lines 10-15 with new code
  replace_lines(file_path="src/auth.py", content="    def verify(self):\\n        return True\\n", start_line=10, end_line=15)

  # Delete lines 20-25
  replace_lines(file_path="src/old.py", content="", start_line=20, end_line=25)

  # Insert after line 5 (without removing anything)
  replace_lines(file_path="src/main.py", content="import os\\n", start_line=6, end_line=5)

WORKFLOW:
  1. read_file(file_path="src/foo.py")   → see numbered lines
  2. replace_lines(file_path="src/foo.py", content="new code\\n", start_line=10, end_line=20)""",

    "edit_symbol": """\
edit_symbol(symbol, new_code, file_path?)

Replace an entire method or function by symbol name. Uses the code index — no text matching.

PARAMS:
  symbol (str, required)     — Qualified name: "ClassName.method_name" or "function_name"
  new_code (str, required)   — Complete new source (including def/async def line)
  file_path (str, optional)  — File hint if symbol name is ambiguous

EXAMPLES:
  edit_symbol(symbol="AuthService.verify_token", new_code="    def verify_token(self, token):\\n        payload = self._decode(token):\\n        return payload if payload else None\\n")

  edit_symbol(symbol="parse_config", new_code="def parse_config(path):\\n    with open(path) as f:\\n        return json.load(f)\\n")

TIPS:
  - Indentation is auto-adjusted to match the original
  - If the symbol is ambiguous, provide file_path to disambiguate
  - Use get_symbol_code first to see the current implementation""",

    "add_symbol": """\
add_symbol(file_path, code, class_name?)

Add a new method or function to a file or class.

PARAMS:
  file_path (str, required)   — Target file
  code (str, required)        — Complete method/function source (including def line)
  class_name (str, optional)  — Class to add to (indentation auto-adjusted). If empty, appends to end of file.

EXAMPLES:
  # Add method to a class
  add_symbol(file_path="src/auth.py", code="def validate(self):\\n    return bool(self.token)\\n", class_name="AuthService")

  # Add standalone function to file
  add_symbol(file_path="src/utils.py", code="def helper():\\n    pass\\n")""",

    "remove_symbol": """\
remove_symbol(symbol, file_path?)

Remove a method or function by name.

PARAMS:
  symbol (str, required)     — Qualified name: "ClassName.method_name" or "function_name"
  file_path (str, optional)  — File hint if ambiguous

EXAMPLES:
  remove_symbol(symbol="AuthService._deprecated_method")
  remove_symbol(symbol="old_helper_function", file_path="src/utils.py")""",

    "get_symbol_code": """\
get_symbol_code(name, file_path?)

Get the full source code of a function, method, or class by name.

PARAMS:
  name (str, required)       — Qualified name: "ClassName.method" or "function_name"
  file_path (str, optional)  — File hint for disambiguation

RETURNS: File path, line range, and complete source code.

EXAMPLES:
  get_symbol_code(name="AuthService.verify_token")
  get_symbol_code(name="parse_config", file_path="src/config.py")""",

    "list_symbols": """\
list_symbols(file_path, kind?)

List all symbols defined in a file.

PARAMS:
  file_path (str, required)  — File to list symbols from
  kind (str, optional)       — Filter: "class", "method", "function", "variable"

EXAMPLES:
  list_symbols(file_path="src/auth.py")
  list_symbols(file_path="src/auth.py", kind="method")""",

    "search_symbols": """\
search_symbols(query, kind?)

Search for symbols by name across the entire project. Supports partial matching.

PARAMS:
  query (str, required)  — Symbol name to search (partial match: "token" finds "verify_token")
  kind (str, optional)   — Filter by kind: "class", "method", "function"

EXAMPLES:
  search_symbols(query="verify_token")
  search_symbols(query="Auth", kind="class")""",

    "find_references": """\
find_references(name, ref_kind?)

Find all places where a symbol is used in the codebase.

PARAMS:
  name (str, required)       — Symbol name to find references for
  ref_kind (str, optional)   — Reference kind filter: "usage", "call", "import"

EXAMPLES:
  find_references(name="verify_token")
  find_references(name="AuthService", ref_kind="import")""",

    "project_structure": """\
project_structure(file_path?, depth?)

Show project directory tree with descriptions of what each file contains.

PARAMS:
  file_path (str, optional) — Root path to show. Default: workspace root.
  depth (int, optional)     — Tree depth. Default: 2.

EXAMPLES:
  project_structure()
  project_structure(file_path="src/", depth=3)""",

    "add_content_after_line": """\
add_content_after_line(file_path, line_number, content)

Insert content AFTER a specific line. The existing line is not modified.
Always read_file first to see line numbers.

PARAMS:
  file_path (str, required)   — Path to the file
  line_number (int, required) — Line to insert after (1-based). Use 0 to insert at the very beginning.
  content (str, required)     — Content to insert

EXAMPLES:
  # Add an import after line 2
  add_content_after_line(file_path="src/main.py", line_number=2, content="import os\\n")

  # Add a method after line 50
  add_content_after_line(file_path="src/auth.py", line_number=50, content="    def logout(self):\\n        self.token = None\\n")

  # Insert at the very beginning of the file
  add_content_after_line(file_path="src/main.py", line_number=0, content="#!/usr/bin/env python\\n")""",

    "add_content_before_line": """\
add_content_before_line(file_path, line_number, content)

Insert content BEFORE a specific line. The existing line is pushed down.
Always read_file first to see line numbers.

PARAMS:
  file_path (str, required)   — Path to the file
  line_number (int, required) — Line to insert before (1-based)
  content (str, required)     — Content to insert

EXAMPLES:
  # Add a comment before line 10
  add_content_before_line(file_path="src/auth.py", line_number=10, content="# TODO: refactor this\\n")

  # Add imports before the first function definition
  add_content_before_line(file_path="src/utils.py", line_number=5, content="from typing import Optional\\n")""",

    "apply_patch": """\
apply_patch(patch, strip?)

Apply a unified diff patch to one or more files. Like running `patch -p1`.

PARAMS:
  patch (str, required) — Unified diff string (like output of `git diff`).
                          Must include diff headers (--- a/file, +++ b/file) and hunks (@@ ... @@).
  strip (int, optional) — Number of leading path components to strip (like `patch -pN`). Default 1.

EXAMPLES:
  # Single file change
  apply_patch(patch=\"\"\"--- a/src/auth.py
+++ b/src/auth.py
@@ -10,3 +10,4 @@
     def verify(self):
-        return True
+        if not self.token:
+            return False
+        return self._check(self.token)
\"\"\")

  # Multi-file change in one call
  apply_patch(patch=\"\"\"--- a/src/auth.py
+++ b/src/auth.py
@@ -1,3 +1,4 @@
 import os
+import jwt
 from .base import Base
--- a/src/config.py
+++ b/src/config.py
@@ -5,1 +5,1 @@
-TIMEOUT = 30
+TIMEOUT = 60
\"\"\")

TIPS:
  - Use for multi-file changes when you can express the fix as a diff
  - Context lines (unchanged) help locate the right position
  - strip=0 if paths are already correct (no a/ b/ prefix)""",

    "git_status": """\
git_status()

Show the current working tree status (staged, unstaged, untracked files).

PARAMS: None

EXAMPLES:
  git_status()""",

    "git_diff": """\
git_diff(branch?, file?, staged?)

Show uncommitted changes or diff against a branch.

PARAMS:
  branch (str, optional)  — Branch or ref to diff against (e.g. "main"). Default: working tree changes.
  file (str, optional)    — Limit diff to a specific file path.
  staged (bool, optional) — If True, show only staged changes. Default: False (unstaged).

EXAMPLES:
  git_diff()
  git_diff(branch="main")
  git_diff(file="src/auth.py")
  git_diff(staged=True)""",

    "git_branch": """\
git_branch(branch_name, create?, base_branch?)

Create or checkout a git branch.

PARAMS:
  branch_name (str, required)  — Name of the branch
  create (bool, optional)      — Create the branch if True (default), checkout if False.
  base_branch (str, optional)  — Base branch to create from. Default: "main".

EXAMPLES:
  git_branch(branch_name="fix-auth", create=True)
  git_branch(branch_name="existing-branch", create=False)
  git_branch(branch_name="feature-x", base_branch="develop")""",

    "git_commit": """\
git_commit(message, files?)

Commit changes to git.

PARAMS:
  message (str, required)       — Commit message
  files (list[str], optional)   — Specific files to commit. Default: all staged changes.

EXAMPLES:
  git_commit(message="Fix auth expiry check")
  git_commit(message="Update config", files=["src/config.py"])""",

    "execute_command": """\
execute_command(command, cwd?, timeout?, env?)

Run a shell command and return its output.

PARAMS:
  command (str, required)           — Shell command to run
  cwd (str, optional)               — Working directory. Default: workspace root.
  timeout (int, optional)           — Timeout in seconds. Default: 300.
  env (dict[str, str], optional)    — Extra environment variables to set.

EXAMPLES:
  execute_command(command="python -m pytest tests/ -x -q")
  execute_command(command="npm test", cwd="frontend/")
  execute_command(command="cargo build", timeout=600)""",

    "code_interpreter": """\
code_interpreter(code, libraries_used?, timeout?)

Execute Python code snippets in an isolated subprocess. Beyond plain
data analysis, this tool gives you read-only access to the project's
code intelligence index — a curated set of query functions is
pre-imported automatically so your script can call them directly
without any import statement.

PARAMS:
  code (str, required)                — Python code to execute
  libraries_used (list[str], optional) — Libraries to make available
  timeout (int, optional)              — Timeout seconds. Default: 120.

PRE-IMPORTED BRIDGE FUNCTIONS (read-only, safe):

  find_symbols(query, kind="", limit=50) -> list[dict]
    FTS5 name search. Returns matched symbols as dicts with
    name, qualified_name, kind, file_path, line_start, line_end,
    signature, docstring, parent_symbol, visibility, language.

  find_definitions(name, kind="") -> list[dict]
    Accepts bare names ("connectToVm") OR dotted names
    ("VirtioSocketWatcherService.connectToVm"). Returns every
    place the symbol is defined.

  find_references(name, file_path="", ref_kind="", limit=200) -> list[dict]
    Every call/usage of a symbol. ref_kind="call" for method
    invocations, "usage" for reads, "" for all.

  list_file_symbols(file_path, kind="") -> list[dict]
    Every symbol defined in one file. file_path can be relative
    to the workspace or absolute.

  iter_symbols(kind="", parent="", language="", file_path="",
               limit=5000) -> list[dict]
    Direct SELECT on ci_symbols — use this when you want to WALK
    all symbols and filter in Python. Each filter is optional.
    Ideal for "count methods per class", "find all TS functions",
    "list every method of class Foo". Do NOT use find_symbols
    with an empty query — that's FTS5 and requires a search term.

  get_source(qualified_name, file_path="") -> str
    Source code of a symbol as a numbered string. Empty string
    if not found.

  find_similar(qualified_name, file_path="", threshold=0.7,
               limit=10) -> list[dict]
    Jaccard-based body similarity. Results include a similarity
    score in [0, 1] and an is_exact_dup flag for copy-paste.

  search_by_intent(query, kind="", limit=20) -> list[dict]
    Natural-language search over docstrings + signatures via
    FTS5 BM25. Use when you know what the code should DO but
    not what it's called. Each result has a "bm25" score
    (lower = better match).

  extract_skeleton(file_path) -> dict
    Tree-sitter structural skeleton of a file. Returns
    {file_path, language, total_lines, total_bytes, symbols}.

  list_files(language="") -> list[str]
    Every indexed file. Filter by language ("typescript", "python",
    etc.) or leave empty for all.

  project_stats() -> dict
    Summary: total_files, total_symbols, symbols_by_kind,
    total_references, total_method_bodies, languages. Good first
    call in any analysis script to orient yourself.

EXAMPLES:

  # Plain Python — no bridge functions
  code_interpreter(code="print(2 + 2)")

  # "How many methods per class?"
  code_interpreter(code=\"\"\"
from collections import Counter
methods = iter_symbols(kind="method")
by_class = Counter(m["parent_symbol"] for m in methods if m["parent_symbol"])
for name, n in by_class.most_common(10):
    print(f"{n:3}  {name}")
\"\"\")

  # "Which methods call connect but never disconnect?"
  code_interpreter(code=\"\"\"
connect_callers = {
    f"{r['file_path']}:{r['line']}" for r in find_references("connect", ref_kind="call")
}
disconnect_callers = {
    f"{r['file_path']}:{r['line']}" for r in find_references("disconnect", ref_kind="call")
}
lifecycle_bugs = connect_callers - disconnect_callers
print(f"{len(lifecycle_bugs)} callers of connect without a matching disconnect")
for loc in list(lifecycle_bugs)[:10]:
    print(f"  {loc}")
\"\"\")

  # "What does ResponseMessage look like, and who uses it?"
  code_interpreter(code=\"\"\"
defs = find_definitions("ResponseMessage")
print(f"Defined in {len(defs)} place(s):")
for d in defs:
    print(f"  {d['file_path']}:{d['line_start']}")
refs = find_references("ResponseMessage", limit=20)
print(f"Referenced {len(refs)} times across the project")
\"\"\")

  # "Find duplicate method bodies in the codebase"
  code_interpreter(code=\"\"\"
# pick any method you want to compare against
matches = find_similar("VirtioSocketWatcherService.connectToVm", threshold=0.8)
for m in matches:
    tag = "EXACT" if m["is_exact_dup"] else f"{m['similarity']:.0%}"
    print(f"[{tag}] {m['qualified_name']} @ {m['file_path']}:{m['line_start']}")
\"\"\")

TIPS:

  * Call project_stats() first in any analysis script to see
    what's in the index before querying.
  * iter_symbols is the RIGHT tool for "walk all X". find_symbols
    requires a search query.
  * All bridge functions return plain dicts. JSON-serialise them
    with json.dumps() if you want to print tables.
  * The bridge is read-only — it cannot mutate anything. To edit
    files, exit the script and use create_file/replace_lines.""",

    "record_finding": """\
record_finding(title, content, finding_type?, confidence?, tags?, sources?)

Save a finding to the knowledge base for future sessions.

PARAMS:
  title (str, required)             — Searchable title/topic
  content (str, required)           — Finding details (self-contained, useful without extra context)
  finding_type (str, optional)      — "observation" (default), "conclusion", or "project_context"
  confidence (float, optional)      — 0.0-1.0. Default: 0.5. Use 0.8+ for verified facts.
  tags (list[str], optional)        — Tags for categorization
  sources (list[str], optional)     — Source URLs or file paths

EXAMPLES:
  record_finding(title="Auth module structure", content="JWT with HS256, verify_token at src/auth.py:42", finding_type="project_context", confidence=0.9)
  record_finding(title="Redis 7.2 caching", content="Supports client-side caching via RESP3", sources=["https://redis.io/docs/"])""",

    "read_findings": """\
read_findings(query?, finding_type?, limit?)

Read stored findings from the knowledge base.

PARAMS:
  query (str, optional)         — Filter findings by text match
  finding_type (str, optional)  — Filter by type: "observation", "conclusion", "project_context"
  limit (int, optional)         — Max results. Default: 50.

EXAMPLES:
  read_findings()
  read_findings(query="auth")
  read_findings(finding_type="project_context", limit=10)""",

    "search_findings": """\
search_findings(query, limit?)

Semantic search over findings using embeddings. Better than read_findings for fuzzy queries.

PARAMS:
  query (str, required)   — Search query (semantic matching, not exact text)
  limit (int, optional)   — Max results. Default: 20.

EXAMPLES:
  search_findings(query="authentication flow")
  search_findings(query="database connection pooling", limit=5)""",

    "search_knowledge": """\
search_knowledge(query, sources?, limit?)

Unified search across findings, reports, and cached documentation.

PARAMS:
  query (str, required)           — Search query
  sources (list[str], optional)   — Which sources to search. Default: ["findings", "reports"].
  limit (int, optional)           — Max results. Default: 20.

EXAMPLES:
  search_knowledge(query="rate limiting")
  search_knowledge(query="FastAPI auth", sources=["findings", "reports"])""",

    "web_search": """\
web_search(query, num_results?)

Search the web for documentation, APIs, error messages, or solutions.

PARAMS:
  query (str, required)       — Search query (be specific for better results)
  num_results (int, optional) — Number of results. Default: 10.

EXAMPLES:
  web_search(query="python requests timeout configuration")
  web_search(query="FastAPI dependency injection lifespan", num_results=5)""",

    "web_fetch": """\
web_fetch(url, format?)

Fetch and read a web page. Prefer official documentation URLs.

PARAMS:
  url (str, required)            — URL to fetch
  format (str, optional)         — "markdown" (default) or "text". Markdown preserves structure.

EXAMPLES:
  web_fetch(url="https://docs.python.org/3/library/json.html")
  web_fetch(url="https://api.example.com/docs", format="text")""",

    "rename_symbol": """\
rename_symbol(symbol, new_name, file_path?)

Rename a symbol and update ALL references and imports across the entire project.

PARAMS:
  symbol (str, required)     — Qualified name: "ClassName.method_name" or "function_name"
  new_name (str, required)   — New name (just the name, not qualified)
  file_path (str, optional)  — File hint if symbol name is ambiguous

EXAMPLES:
  rename_symbol(symbol="verify_token", new_name="validate_token")
  rename_symbol(symbol="AuthService.check", new_name="authenticate", file_path="src/auth.py")
  rename_symbol(symbol="UserModel", new_name="User")

WHAT IT DOES:
  1. Renames the definition (def/class line)
  2. Updates all references in every file (calls, usages)
  3. Updates all import statements
  4. Re-indexes all modified files""",

    "move_symbol": """\
move_symbol(symbol, target_file, target_class?, after_line?)

Move a function, method, or class to another file or into a class. Updates imports project-wide.

PARAMS:
  symbol (str, required)       — What to move: "ClassName.method", "function_name", or "ClassName"
  target_file (str, required)  — Destination file path
  target_class (str, optional) — Class to move into. Empty = top-level in file.
  after_line (int, optional)   — Insert after this line (1-based). 0 = end of file/class.

EXAMPLES:
  # Move function to another file (at the end)
  move_symbol(symbol="validate_input", target_file="src/validators.py")

  # Move method into a class
  move_symbol(symbol="helper_func", target_file="src/service.py", target_class="UserService")

  # Move class to new file, insert after line 10
  move_symbol(symbol="UserModel", target_file="src/models/user.py", after_line=10)

WHAT IT DOES:
  1. Extracts the symbol's source code from the original file
  2. Removes it from the source (cleans up blank lines)
  3. Inserts into the target (with correct indentation if target_class)
  4. Updates import statements in all files that referenced the symbol
  5. Re-indexes both source and target files""",

    "step_complete": """\
step_complete(summary, status, final_answer?)

Signal that the current step is complete. You MUST call this after each step.

PARAMS:
  summary (str, required)       — Structured summary (~150 tokens): "Read: ... | Changed: ... | Remaining: ..."
  status (str, required)        — "continue" (more work), "done" (task complete), "blocked" (stuck)
  final_answer (str, required when status=done) — Final user-facing result

To manage the plan, use add_step/modify_step/remove_step BEFORE calling step_complete.

EXAMPLES:
  add_step(title="Fix verify_token() expiry check in auth.py:42",
    explanation="Use edit_symbol to add datetime.utcnow() comparison against exp field.")
  add_step(title="Run pytest tests/test_auth.py to verify fix")
  step_complete(summary="Read: auth.py — found verify_token() on line 42", status="continue")

  step_complete(summary="Changed: auth.py:42 — added expiry check", status="done",
    final_answer="Fixed the token expiry bug in auth.py.")

IMPORTANT:
  - Before calling, save key findings via add_note (they survive between steps)
  - Before status="done", call add_session_note (persists across tasks)
  - Use add_step/modify_step/remove_step to manage the plan, NOT step_complete""",

    "add_step": """\
add_step(title, explanation?, index?)

Add a new step to the plan WITHOUT completing the current step.
Use when you discover new work mid-step. Omit index to append at end.

PARAMS:
  title (str, required)           — Short title: FILE, FUNCTION, CHANGE
  explanation (str, optional)     — Detailed explanation of how to approach the step
  index (int, optional)           — Step number. Omit to append at end of plan.

EXAMPLES:
  add_step(title="Run pytest tests/test_auth.py to verify fix")
  add_step(title="Fix refresh_token() in auth.py:85", index=4)""",

    "modify_step": """\
modify_step(index, title?, explanation?)

Modify the title or explanation of a pending step WITHOUT completing the current step.

PARAMS:
  index (int, required)           — Step number to modify
  title (str, optional)           — New title (empty = keep current)
  explanation (str, optional)     — New explanation (empty = keep current)

EXAMPLE:
  modify_step(index=4, title="Also fix refresh_token() in auth.py:85")""",

    "remove_step": """\
remove_step(index)

Remove a pending step from the plan WITHOUT completing the current step.

PARAMS:
  index (int, required) — Step number to remove

EXAMPLE:
  remove_step(index=6)""",

    "analyze_code": """\
analyze_code(file_path?, checks?)

Run heuristic analysis to detect code errors using indexed data. Very fast — no re-parsing.

PARAMS:
  file_path (str, optional)  — File to analyze. Empty = whole project.
  checks (str, optional)     — Comma-separated: broken_imports, undefined_symbols, unused_imports, unused_definitions. Empty = all.

CHECKS:
  broken_imports      — Imports that can't be resolved (error severity)
  undefined_symbols   — References to symbols with no definition (warning)
  unused_imports      — Imports never referenced in the same file (warning)
  unused_definitions  — Symbols defined but never referenced anywhere (hint)

EXAMPLES:
  analyze_code(file_path="src/auth.py")
  analyze_code(checks="broken_imports,unused_imports")
  analyze_code()  # full project scan

RETURNS: JSON with errors, warnings, hints grouped by severity.""",
}


class HelpTool(InfinibayBaseTool):
    name: str = "help"
    description: str = "Get detailed help and examples for any tool."
    args_schema: Type[BaseModel] = HelpInput

    def _run(self, context: str | None = None) -> str:
        if context is not None:
            context = context.strip().lower()

        # Direct match
        if context in HELP_CONTENT:
            return HELP_CONTENT[context]

        # Try matching as category index key
        if context in _CATEGORY_INDEX:
            return HELP_CONTENT.get(context, f"No help available for category: {context}")

        # Fuzzy: search for context as substring in keys
        matches = [k for k in HELP_CONTENT if k and context and context in k]
        if len(matches) == 1:
            return HELP_CONTENT[matches[0]]
        if len(matches) > 1:
            return f"Multiple matches for '{context}': {', '.join(matches)}. Be more specific."

        available = sorted(k for k in HELP_CONTENT if k is not None)
        return (
            f"No help found for '{context}'.\n"
            f"Available topics: {', '.join(available)}"
        )
