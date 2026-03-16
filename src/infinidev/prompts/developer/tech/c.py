"""C-specific coding guidelines for the Developer agent."""


def get_prompt() -> str:
    return """\
## C Guidelines

### Naming Conventions
- `snake_case` for functions and variables.
- `UPPER_SNAKE_CASE` for macros and constants.
- `PascalCase` for `typedef struct` type names.
- Prefix all public API symbols with a module prefix (e.g. `mylib_init()`, `mylib_free()`).

### Error Handling
- Return error codes (0 for success, negative for error) or `NULL` for pointer-returning functions.
- Always check return values of `malloc`, `fopen`, `read`, `write`, and all system calls.
- Use `errno` with `perror()` or `strerror(errno)` for system error reporting.
- Free resources in reverse allocation order; use `goto cleanup` pattern for complex functions.

### Security
- Use `snprintf` not `sprintf` — always bound buffer writes.
- Use `strncpy` / `strncat` not `strcpy` / `strcat`.
- Never use `gets()` — use `fgets()` instead.
- Validate all array indices before access.
- Use `calloc` to zero-initialise allocated memory.
- Prevent format string attacks: `printf("%s", user_input)` not `printf(user_input)`.

### Standard Library / Core Tooling
- `<string.h>` for string operations.
- `<stdlib.h>` for memory allocation and conversions.
- `<stdio.h>` for I/O.
- `<errno.h>` for error codes.
- `<stdint.h>` for fixed-width integer types (`uint32_t`, `int64_t`).
- `<stdbool.h>` for `bool`, `true`, `false`.
- `<assert.h>` for debug-only assertions.

### Useful Patterns
- Use `typedef struct { ... } TypeName;` for opaque data types.
- Use `const` for read-only pointer parameters (`const char *str`).
- Use `static` for file-scope (internal linkage) functions and variables.
- Use `#ifndef HEADER_H` / `#define HEADER_H` include guards in all headers.
- Use `sizeof(*ptr)` in `malloc` calls instead of `sizeof(Type)` for maintainability.

### Anti-Patterns to Avoid
- Variable-length arrays (VLAs) in security-sensitive code — use heap allocation.
- Implicit `int` return type — always declare return types explicitly.
- Mixing signed and unsigned comparisons — use consistent types.
- `realloc` without saving the old pointer — if `realloc` fails the original memory leaks.
- `free(ptr)` without setting `ptr = NULL` afterward — risks use-after-free.\
"""
