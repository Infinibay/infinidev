"""TypeScript-specific coding guidelines for the Developer agent."""


def get_prompt() -> str:
    return """\
## TypeScript Guidelines

### Naming Conventions
- `PascalCase` for types, interfaces, classes, and enums.
- `camelCase` for variables, functions, and method names.
- `SCREAMING_SNAKE_CASE` for module-level constants.
- `kebab-case` for file names (e.g. `user-service.ts`).

### Error Handling
- Prefer typed `Result<T, E>` patterns or discriminated unions over `try/catch` for expected failures.
- Reserve `try/catch` for truly exceptional I/O errors.
- Always narrow `unknown` in catch blocks before accessing properties:
  ```ts
  catch (err: unknown) {
    if (err instanceof SomeError) { /* handle */ }
  }
  ```

### Security
- Never use `as any` to bypass type checks at trust boundaries — validate external data with a schema library (e.g. `zod`).
- Avoid `eval`, `Function()`, and `innerHTML` assignment.
- Sanitise all user-supplied strings before embedding in HTML or SQL.

### Standard Library / Core Tooling
- Always enable `tsc --strict`.
- Use `ts-node` or `tsx` for running scripts directly.
- Install `@types/*` packages for third-party libraries that lack built-in types.
- Prefer Node stdlib modules: `path`, `fs/promises`, `crypto`, `url`.

### Useful Patterns
- Use the `satisfies` operator for type-safe object literals without widening.
- Prefer `readonly` arrays and `Readonly<T>` for immutable data.
- Use `as const` assertions for literal types and exhaustive switch checks.
- Use discriminated unions with a `type` or `kind` field for variant modelling.

### Anti-Patterns to Avoid
- `any` casts at API boundaries — use proper validation instead.
- Ignoring `Promise` rejections — always attach `.catch()` or use `try/await`.
- `!` non-null assertions without a comment explaining why the value is guaranteed to exist.
- `namespace` merging in new code — use ES modules instead.
- `enum` with computed values — prefer `as const` objects for string enums.\
"""
