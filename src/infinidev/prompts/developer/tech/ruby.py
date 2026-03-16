"""Ruby-specific coding guidelines for the Developer agent."""


def get_prompt() -> str:
    return """\
## Ruby Guidelines

### Naming Conventions
- `snake_case` for methods, variables, and file names.
- `PascalCase` for classes and modules.
- `SCREAMING_SNAKE_CASE` for constants.
- `?` suffix for predicate methods that return a boolean (e.g. `empty?`, `valid?`).
- `!` suffix for mutating or dangerous methods (e.g. `save!`, `sort!`).

### Error Handling
- Rescue specific exception classes — never bare `rescue` without a class.
- Use `ensure` for cleanup that must run regardless of success or failure.
- Raise `StandardError` subclasses with descriptive messages.
- Use `begin / rescue / ensure / end` blocks; avoid inline `rescue` on assignments.

### Security
- Use parameterised queries — `ActiveRecord` placeholders or `pg` gem with `$1` bindings.
- Avoid `eval`, `send`, and `public_send` with user-controlled input.
- Never use `system` / `exec` / backticks with interpolated user strings — use `Open3.capture3` with array arguments.
- Use `SecureRandom.hex` or `SecureRandom.uuid` for tokens — never `rand`.

### Standard Library / Core Tooling
- `Pathname` for path manipulation.
- `FileUtils` for file/directory operations.
- `JSON`, `CSV` for data formats.
- `Logger` for structured logging.
- `Digest`, `OpenSSL` for cryptography.
- `Net::HTTP` for HTTP requests (or use `Faraday` / `HTTParty` gems).
- `Tempfile`, `StringIO` for temporary and in-memory I/O.

### Useful Patterns
- Use `Enumerable` methods (`map`, `select`, `reduce`, `each_with_object`) over manual loops.
- Use `Struct` or `Data` (Ruby 3.2+) for lightweight value objects.
- Use `freeze` on string literals in performance-critical paths.
- Use `Module#prepend` over `alias_method` for method wrapping.

### Anti-Patterns to Avoid
- `rescue Exception` — this catches `SignalException` and `SystemExit`; rescue `StandardError` instead.
- `method_missing` without implementing `respond_to_missing?`.
- Monkey-patching core classes in libraries — use refinements if needed.
- `attr_accessor` for everything — use `attr_reader` when mutation is not needed.
- Deeply nested conditionals — use guard clauses and early returns.\
"""
