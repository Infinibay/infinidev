"""C++-specific coding guidelines for the Developer agent."""


def get_prompt() -> str:
    return """\
## C++ Guidelines

### Naming Conventions
- `UPPER_SNAKE_CASE` for macros and preprocessor constants.
- `PascalCase` for classes, structs, enums, and type aliases.
- `camelCase` or `snake_case` for methods and local variables — pick one convention and enforce it project-wide.
- `m_` prefix for private member variables (e.g. `m_count`).
- `k` prefix for compile-time constants (e.g. `kMaxRetries`).

### Error Handling
- Use exceptions for truly exceptional conditions only — not for expected control flow.
- Use `std::optional<T>` for values that may be absent.
- Use `std::expected<T, E>` (C++23) or `std::variant` for result-or-error return types.
- RAII for all resource management — constructors acquire, destructors release.

### Security
- Prefer smart pointers (`std::unique_ptr`, `std::shared_ptr`) over raw `new` / `delete`.
- Use `std::string_view` for read-only string parameters.
- Avoid C-style casts — use `static_cast`, `dynamic_cast`, `reinterpret_cast` explicitly.
- Never use `gets`, `scanf` without width limits, or `sprintf`.
- Avoid `reinterpret_cast` and `const_cast` without a safety comment.

### Standard Library / Core Tooling
- `<algorithm>` and `<ranges>` (C++20) for data transformations.
- `<memory>` for smart pointers.
- `<string>`, `<string_view>` for text.
- `<vector>`, `<array>`, `<unordered_map>`, `<map>` for containers.
- `<filesystem>` for path and file operations.
- `<thread>`, `<mutex>`, `<atomic>` for concurrency.
- `<span>` (C++20) for non-owning contiguous views.

### Useful Patterns
- Follow the Rule of Zero — let the compiler generate special member functions via smart pointers and standard containers.
- Use `[[nodiscard]]` on functions whose return value must be checked.
- Use `constexpr` for compile-time computation.
- Prefer range-based `for` loops over index-based iteration.
- Use structured bindings (`auto [key, value] = ...`) for readability.

### Anti-Patterns to Avoid
- Raw owning pointers — use `unique_ptr` or `shared_ptr`.
- `using namespace std;` in header files — pollutes the global namespace.
- `reinterpret_cast` without a comment explaining the safety invariant.
- `const_cast` to remove const — redesign the API instead.
- Manual memory management (`new` / `delete`) when RAII containers or smart pointers apply.\
"""
