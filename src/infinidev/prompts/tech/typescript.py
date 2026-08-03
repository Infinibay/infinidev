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
- An expected failure is a typed `Result<T, E>` or a discriminated union, never a `try/catch`.
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

### Standard Library / Core Tooling
- Always enable `tsc --strict`.
- Use `ts-node` or `tsx` for running scripts directly.
- Install `@types/*` packages for third-party libraries that lack built-in types.
- Reach for the Node stdlib first: `path`, `fs/promises`, `crypto`, `url`.

### Useful Patterns
- Use the `satisfies` operator for type-safe object literals without widening.
- Immutable data is a `readonly` array or a `Readonly<T>`.
- Use `as const` assertions for literal types and exhaustive switch checks.
- Use discriminated unions with a `type` or `kind` field for variant modelling.

### Anti-Patterns to Avoid
- `any` casts at API boundaries — validate input against a runtime schema
  before narrowing the type.
- Ignoring `Promise` rejections — always attach `.catch()` or use `try/await`.
- `!` non-null assertions without a comment explaining why the value is guaranteed to exist.
- `namespace` merging in new code — use ES modules instead.
- `enum` with computed values — a string enum is an `as const` object.\
"""
