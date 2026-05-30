# Repository Guidelines

## Project Structure & Module Organization

Infinidev is a Python 3.13+ terminal AI programming tool. Core package code lives under `src/infinidev/`: `cli/` contains the TUI and command entrypoint, `engine/` owns orchestration and agent loops, `tools/` contains tool implementations, `code_intel/` handles indexing and symbol lookup, and `ui/` contains Textual widgets and handlers. Tests live in `tests/`, with integration and interactive cases under `tests/integration/` and `tests/interactive/`. Supporting areas include `docs/` for design notes, `bench/` for evaluation scripts, `finetune/` for dataset/training utilities, `scripts/` for one-off helpers, `examples/` for sample projects, and `public/` for static assets such as screenshots.

## Build, Test, and Development Commands

- `uv sync`: install project and development dependencies from `pyproject.toml` and `uv.lock`.
- `uv run infinidev`: launch the default TUI.
- `uv run infinidev --classic`: launch the text-only renderer.
- `uv run pytest`: run the configured test suite in `tests/`.
- `uv run pytest tests/test_config.py -v`: run a focused test file with verbose output.
- `./install.sh`: install the wrapper for local system use.

## Coding Style & Naming Conventions

Use idiomatic Python with 4-space indentation, type hints where they clarify contracts, and small modules aligned with the existing package boundaries. Name files and functions in `snake_case`, classes in `PascalCase`, and constants/settings in `UPPER_SNAKE_CASE`. Keep tool implementations in the appropriate `src/infinidev/tools/<category>/` package and prefer existing helpers over new abstractions. Add comments only for non-obvious control flow or protocol constraints.

## Testing Guidelines

The suite uses `pytest` with `pytest-asyncio` configured in auto mode. Add tests next to related coverage in `tests/` using the `test_*.py` and `test_*` naming patterns. For engine, tool, parser, or UI behavior, include focused regression tests before broader integration checks. Use `uv run pytest tests/path.py::test_name -v` while iterating, then run `uv run pytest` before submitting.

## Commit & Pull Request Guidelines

Recent history uses short imperative or release-style subjects, for example `hooks: wire on_file_change...`, `Fix two issues...`, and `Release 0.6.0: ...`. Keep commits focused and describe the user-visible behavior or subsystem changed. Pull requests should include a concise summary, tests run, linked issues when applicable, and screenshots or terminal output for TUI-visible changes.

## Security & Configuration Tips

Project-local runtime state may appear in `.infinidev/`; do not commit secrets, generated databases, logs, model outputs, or large finetune artifacts. Configuration is controlled by `.infinidev/settings.json` and `INFINIDEV_` environment variables; document any new setting in `README.md` when it affects users.
