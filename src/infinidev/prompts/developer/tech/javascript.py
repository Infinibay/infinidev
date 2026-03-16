"""JavaScript-specific coding guidelines for the Developer agent."""


def get_prompt() -> str:
    return """\
## JavaScript Guidelines

### Naming Conventions
- `camelCase` for variables and functions.
- `PascalCase` for classes and constructor functions.
- `SCREAMING_SNAKE_CASE` for module-level constants.
- Use JSDoc `@param` / `@returns` annotations for public APIs.

### Error Handling
- Always handle Promise rejections with `.catch()` or `try/await`.
- Never swallow errors silently — at minimum log them.
- Use `Error` subclasses with a `code` property for programmatic handling:
  ```js
  class NotFoundError extends Error {
    constructor(message) { super(message); this.code = 'NOT_FOUND'; }
  }
  ```

### Security
- Use `textContent` not `innerHTML` for DOM text insertions.
- Use `crypto.randomUUID()` not `Math.random()` for identifiers.
- Avoid `eval` and `new Function` — they enable code injection.
- Sanitise all user input before rendering or embedding in queries.

### Standard Library / Core Tooling
- Modern globals available without imports: `URL`, `URLSearchParams`, `fetch`, `crypto`, `structuredClone`, `AbortController`.
- Use `Array.isArray()` for type checking, not `instanceof Array`.
- Use `Number.isFinite()` and `Number.isNaN()` over their global counterparts.

### Useful Patterns
- Use `Array.from()` over spread for iterables that aren't arrays.
- Use `Object.freeze()` for config constants that must not be mutated.
- Prefer optional chaining `?.` and nullish coalescing `??` over `||` for default values.
- Use `Object.hasOwn(obj, key)` over `obj.hasOwnProperty(key)`.

### Anti-Patterns to Avoid
- `var` declarations — always use `const` or `let`.
- `==` instead of `===` — loose equality causes subtle bugs.
- Mutating function arguments — clone first if mutation is needed.
- Using the `arguments` object — use rest parameters (`...args`) instead.
- Prototype pollution via `Object.assign` with user-controlled input.\
"""
