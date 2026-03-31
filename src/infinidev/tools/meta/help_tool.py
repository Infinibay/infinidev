"""Meta-tool that provides detailed help and examples for all tools."""

from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class HelpInput(BaseModel):
    context: str | None = Field(
        default=None,
        description="Tool name or category to get help for. Omit for overview.",
    )


# ---------------------------------------------------------------------------
# Help content: categories and individual tools
# ---------------------------------------------------------------------------

_CATEGORY_INDEX = {
    "file": ["read_file", "partial_read", "create_file", "replace_lines", "add_content_after_line", "add_content_before_line", "list_directory", "glob", "code_search"],
    "code_intel": ["get_symbol_code", "list_symbols", "search_symbols", "find_references", "project_structure", "analyze_code"],
    "edit": ["edit_symbol", "add_symbol", "remove_symbol", "replace_lines", "add_content_after_line", "add_content_before_line", "rename_symbol", "move_symbol"],
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

  read_file(path)
    Read entire file. Returns numbered lines. Auto-indexes the file for code intelligence.

  partial_read(path, start_line, end_line)
    Read a specific line range (both 1-based, inclusive).

  create_file(path, content)
    Create a new file. FAILS if the file already exists.

  replace_lines(file_path, content, start_line, end_line)
    Replace a range of lines with new content. See help("replace_lines") for details.

  add_content_after_line(file_path, line_number, content)
    Insert content after a specific line. See help("add_content_after_line").

  add_content_before_line(file_path, line_number, content)
    Insert content before a specific line. See help("add_content_before_line").

  list_directory(path)
    List files and directories at a path.

  glob(pattern, path?)
    Find files matching a glob pattern.

  code_search(pattern, path?, include?)
    Search for text/regex patterns across files.

TIP: Always read_file first to get line numbers, then use replace_lines or edit_symbol to edit.""",

    # ── Category: edit ────────────────────────────────────────────────────
    "edit": """\
EDIT TOOLS — Modifying existing files

TWO APPROACHES:

1. SYMBOLIC (preferred for methods/functions):
   edit_symbol(symbol, new_code)  — Replace a method/function by name
   add_symbol(code, file_path)    — Add a method to a file or class
   remove_symbol(symbol)          — Remove a method by name

2. LINE-BASED (for anything else):
   replace_lines(file_path, content, start_line, end_line)
   — Deterministic: specify exact line range, no text matching needed
   — Use read_file first to see line numbers

3. INSERT (add new content without removing):
   add_content_after_line(file_path, line_number, content)
   add_content_before_line(file_path, line_number, content)

4. REFACTORING (project-wide operations):
   rename_symbol(symbol, new_name)  — Rename everywhere: definition + all references + imports
   move_symbol(symbol, target_file) — Move to another file/class, update imports

WORKFLOW:
  1. read_file("src/foo.py")           → see the code with line numbers
  2. edit_symbol("Foo.bar", new_code)  → replace method by name
  OR
  2. replace_lines("src/foo.py", new_code, 10, 25)  → replace lines 10-25""",

    # ── Category: code_intel ──────────────────────────────────────────────
    "code_intel": """\
CODE INTELLIGENCE TOOLS — Symbol-based code navigation

  get_symbol_code(symbol, file_path?)
    Get the full source code of a function, method, or class by name.
    Use qualified names: "ClassName.method_name" or just "function_name".

  list_symbols(file_path, kind?)
    List all symbols in a file. Filter by kind: "class", "method", "function", "variable".

  search_symbols(name, kind?, limit?)
    Search for symbols by name across the project. Supports partial matching.

  find_references(name, kind?, file_path?)
    Find all places where a symbol is used. Essential before renaming or removing.

  project_structure(path?, depth?)
    Show project directory tree.

TIP: Files are auto-indexed when you read_file them. Symbol tools work best after reading.""",

    # ── Category: git ─────────────────────────────────────────────────────
    "git": """\
GIT TOOLS

  git_status()               — Show working tree status
  git_diff(ref?, staged?)    — Show changes (unstaged by default)
  git_branch(name?, action?) — Create, switch, list, or delete branches
  git_commit(message, files?)— Commit changes""",

    # ── Category: shell ───────────────────────────────────────────────────
    "shell": """\
SHELL TOOLS

  execute_command(command, cwd?, timeout?)
    Run a shell command. Use for: running tests, installing packages, build commands.

  code_interpreter(code, language?)
    Execute code snippets (Python by default) in an isolated environment.""",

    # ── Category: knowledge ───────────────────────────────────────────────
    "knowledge": """\
KNOWLEDGE TOOLS — Findings and reports

  record_finding(title, content, tags?)  — Save a finding
  read_findings(tag?, limit?)            — Read stored findings
  search_findings(query)                 — Semantic search over findings
  search_knowledge(query)                — Search across findings + docs""",

    # ── Category: web ─────────────────────────────────────────────────────
    "web": """\
WEB TOOLS

  web_search(query, num_results?)  — Search the web
  web_fetch(url)                   — Fetch content from a URL""",

    # ── Individual tools ──────────────────────────────────────────────────

    "read_file": """\
read_file(path)

Read the full contents of a file. Returns numbered lines for easy reference.
Automatically indexes the file for code intelligence (symbol lookup, etc).

PARAMS:
  path (str, required) — File path (absolute or relative to workspace)

RETURNS: Numbered lines in format "  LINE_NUM\\tCONTENT"

EXAMPLES:
  read_file(file_path="src/auth.py")
  read_file(file_path="tests/test_auth.py")

TIPS:
  - For large files, use partial_read instead to read only what you need
  - Line numbers in the output can be used with replace_lines for editing
  - Binary files are automatically detected and rejected""",

    "partial_read": """\
partial_read(path, start_line, end_line)

Read a specific range of lines from a file. Both bounds are 1-based and inclusive.

PARAMS:
  path (str, required)       — File path
  start_line (int, required) — First line to read (1-based)
  end_line (int, required)   — Last line to read (1-based, inclusive)

EXAMPLES:
  partial_read(file_path="src/auth.py", start_line=10, end_line=30)
  partial_read(file_path="src/main.py", start_line=1, end_line=5)  # just the imports""",

    "create_file": """\
create_file(path, content)

Create a new file. FAILS if the file already exists. Creates parent directories as needed.

PARAMS:
  path (str, required)    — Path for the new file
  content (str, required) — Content to write

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
  edit_symbol(symbol="AuthService.verify_token", new_code="    def verify_token(self, token):\\n        payload = self._decode(token)\\n        return payload if payload else None\\n")

  edit_symbol(symbol="parse_config", new_code="def parse_config(path):\\n    with open(path) as f:\\n        return json.load(f)\\n")

TIPS:
  - Indentation is auto-adjusted to match the original
  - If the symbol is ambiguous, provide file_path to disambiguate
  - Use get_symbol_code first to see the current implementation""",

    "add_symbol": """\
add_symbol(code, file_path, class_name?, position?)

Add a new method or function to a file or class.

PARAMS:
  code (str, required)        — Complete method/function source
  file_path (str, required)   — Target file
  class_name (str, optional)  — Class to add to (indentation auto-adjusted)
  position (str, optional)    — "end" (default) — where to insert

EXAMPLES:
  # Add method to a class
  add_symbol(code="def validate(self):\\n    return bool(self.token)\\n", file_path="src/auth.py", class_name="AuthService")

  # Add standalone function to file
  add_symbol(code="def helper():\\n    pass\\n", file_path="src/utils.py")""",

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
get_symbol_code(symbol, file_path?)

Get the full source code of a function, method, or class by name.

PARAMS:
  symbol (str, required)     — Qualified name: "ClassName.method" or "function_name"
  file_path (str, optional)  — File hint for disambiguation

RETURNS: File path, line range, and complete source code.

EXAMPLES:
  get_symbol_code(symbol="AuthService.verify_token")
  get_symbol_code(symbol="parse_config", file_path="src/config.py")""",

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
search_symbols(name, kind?, limit?)

Search for symbols by name across the entire project. Supports partial matching.

PARAMS:
  name (str, required)   — Symbol name (partial match: "token" finds "verify_token")
  kind (str, optional)   — Filter by kind: "class", "method", "function"
  limit (int, optional)  — Max results (default 10)

EXAMPLES:
  search_symbols(name="verify_token")
  search_symbols(name="Auth", kind="class")""",

    "find_references": """\
find_references(name, kind?, file_path?)

Find all places where a symbol is used in the codebase.

PARAMS:
  name (str, required)       — Symbol name to find references for
  kind (str, optional)       — Reference kind filter: "usage", "call", "import"
  file_path (str, optional)  — Limit search to a specific file

EXAMPLES:
  find_references(name="verify_token")
  find_references(name="AuthService", kind="import")""",

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
