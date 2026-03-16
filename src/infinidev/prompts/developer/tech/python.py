"""Python-specific coding guidelines for the Developer agent."""


def get_prompt() -> str:
    return """\
## Python Guidelines

### Naming Conventions
- `snake_case` for functions, variables, and module names.
- `PascalCase` for classes.
- `UPPER_SNAKE_CASE` for module-level constants.
- `_leading_underscore` for private/internal members.

### Error Handling
- Catch specific exceptions — never bare `except:` or broad `except Exception`.
- Use `contextlib.suppress(SpecificError)` only for truly ignorable errors.
- Always re-raise with `raise ... from err` to preserve the exception chain.
- Use custom exception classes inheriting from a project-level base exception.

### Security
- Use `subprocess` with `shell=False` and a list of arguments — never `shell=True` with user input.
- Use the `secrets` module (not `random`) for tokens, keys, and nonces.
- Parameterise all database queries — never format user input into SQL strings.
- Never `pickle.load` untrusted data — use JSON or a safe serialisation format.

### Standard Library / Core Tooling
- `pathlib.Path` over `os.path` for file system operations.
- `dataclasses` or `attrs` for data containers — prefer `@dataclass(frozen=True)` for value objects.
- `logging` (not `print`) for diagnostics.
- `contextlib`, `itertools`, `functools` for clean utility code.
- Type-annotate all public APIs using the `typing` module.

### Useful Patterns
- Use `__slots__` on performance-critical classes to reduce memory and speed up attribute access.
- Prefer generators and `yield` over building full lists when the consumer iterates once.
- Use `@dataclass(frozen=True)` for immutable value objects.
- Use `Enum` or `StrEnum` for fixed sets of string constants.

### Anti-Patterns to Avoid
- Mutable default arguments (`def f(items=[])`) — use `None` and assign inside.
- `from module import *` — always import explicitly.
- `os.system()` — use `subprocess.run()` instead.
- String formatting for SQL or shell commands — use parameterised queries and argument lists.
- Module-level global mutable state — pass state explicitly or use dependency injection.\
"""
