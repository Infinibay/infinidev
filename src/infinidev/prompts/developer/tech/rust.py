"""Rust-specific coding guidelines for the Developer agent."""


def get_prompt() -> str:
    return """\
## Rust Guidelines

### Naming Conventions
- `snake_case` for functions, variables, and module names.
- `PascalCase` for types, traits, and enum variants.
- `SCREAMING_SNAKE_CASE` for constants and statics.
- `kebab-case` for crate names in `Cargo.toml`.

### Error Handling
- Return `Result<T, E>` for all fallible operations.
- Use the `?` operator for concise error propagation.
- Define domain-specific error enums implementing `std::error::Error` and `Display`.
- Avoid `.unwrap()` in library code — use `.expect("reason")` only in binaries where the invariant is documented.

### Security
- Avoid `unsafe` blocks unless absolutely necessary — every `unsafe` block must have a `// SAFETY:` comment.
- Never use raw pointers without documenting the safety invariant.
- Validate all slice indices and use `.get()` to avoid panics.
- Use the `zeroize` crate for sensitive data (passwords, keys) in memory.

### Standard Library / Core Tooling
- `std::collections::{HashMap, BTreeMap, HashSet}` for data structures.
- `std::sync::{Arc, Mutex, RwLock}` for shared state.
- `std::io::{BufReader, BufWriter}` for efficient I/O.
- `std::path::PathBuf` and `std::fs` for file operations.
- `std::env` for environment variables.

### Useful Patterns
- Use `impl Trait` in return position for flexible API design.
- Use the builder pattern for constructing complex structs.
- Leverage `From` / `Into` traits for clean type conversions.
- Add `#[derive(Debug, Clone, PartialEq)]` liberally on data types.
- Use `#[must_use]` on functions whose return value should not be ignored.

### Anti-Patterns to Avoid
- `.clone()` to silence the borrow checker — understand ownership and borrowing instead.
- `Box<dyn Error>` in library APIs — use concrete error types for better downstream handling.
- `Mutex<Vec<T>>` for read-heavy workloads — use `RwLock` instead.
- `String` parameters where `&str` suffices — accept borrowed data when ownership is not needed.
- Ignoring `clippy` warnings — run `cargo clippy` and address all lints.\
"""
