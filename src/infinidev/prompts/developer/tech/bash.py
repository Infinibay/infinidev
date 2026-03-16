"""Bash/Shell-specific coding guidelines for the Developer agent."""


def get_prompt() -> str:
    return """\
## Bash / Shell Guidelines

### Naming Conventions
- `snake_case` for functions and local variables.
- `UPPER_SNAKE_CASE` for environment variables and exported constants.
- Prefix local variables with `local` inside functions.
- Use descriptive names (`input_file` not `f`, `retry_count` not `n`).

### Error Handling
- Start every script with `set -euo pipefail` to catch errors early.
- Use `trap 'cleanup_function' EXIT` for resource cleanup (temp files, lock files).
- Check exit codes explicitly for commands where `set -e` is insufficient (e.g. inside `if` conditions):
  ```bash
  if ! command_that_may_fail; then
    echo "Error: command failed" >&2
    exit 1
  fi
  ```
- Use `|| { echo "error" >&2; exit 1; }` for inline error handling.

### Security
- Quote all variable expansions: `"$var"` not `$var` — prevents word splitting and glob expansion.
- Use `--` to separate options from arguments in commands that accept it.
- Never `eval` user input or untrusted data.
- Use `mktemp` for temporary files — never hardcode temp file paths.
- Validate inputs with `[[ "$input" =~ ^[a-zA-Z0-9_]+$ ]]` before use.
- Prefer `[[ ]]` over `[ ]` for conditionals — it handles empty variables safely.

### Standard Tooling
- `awk`, `sed`, `grep` for text processing.
- `find` with `-exec` or `-print0 | xargs -0` for safe file iteration.
- `sort`, `uniq`, `cut`, `tr` for data transformation.
- `tee` for writing to both stdout and a file.
- `date`, `printf` (not `echo` for portable output).
- `jq` for JSON processing.

### Useful Patterns
- Use functions for reusable logic — keep the main script body short.
- Use `readonly` for constants that should not be reassigned.
- Use `local` for all variables inside functions to avoid polluting the global scope.
- Use `$(command)` not backticks for command substitution — backticks don't nest.
- Use `[[ -f "$file" ]]` for file existence checks; `[[ -d "$dir" ]]` for directories.

### Anti-Patterns to Avoid
- Unquoted variables — causes word splitting, glob expansion, and security issues.
- Parsing `ls` output — use `find` or shell globs (`for f in *.txt`) instead.
- `cat file | grep pattern` — use `grep pattern file` directly (useless use of cat).
- `cd` without checking success — use `cd /path || exit 1`.
- Scripts without a shebang line (`#!/usr/bin/env bash`) — behaviour is unpredictable without it.\
"""
