# Repository Guidelines

## Project Structure & Module Organization

Infinidev is a Python 3.13+ terminal AI programming tool. Core package code lives under `src/infinidev/`:

- `cli/` — TUI and command entrypoint (`infinidev.cli.main:main`).
- `engine/` — orchestration, agent loops, formats, prompts, analysis (`engine/loop/engine.py` is the developer plan-execute-summarize loop).
- `tools/` — tool implementations grouped by category (`file/`, `code_intel/`, `git/`, `shell/`, `knowledge/`, `web/`, `meta/`, `chat_agent/`, `planner/`, `council/`, `docs/`). Shared base in `tools/base/`.
- `code_intel/` — tree-sitter indexing, symbol lookup, parsers per language.
- `ui/` — Textual widgets, handlers, dialogs.
- `config/`, `db/`, `flows/`, `gather/`, `agents/`, `prompts/` — supporting subsystems.
- `tests/` — unit tests; `tests/integration/` and `tests/interactive/` for higher-level cases. Interactive tests are skipped unless `INFINIDEV_RUN_INTERACTIVE_TESTS=1`.
- `docs/`, `bench/`, `finetune/`, `scripts/`, `examples/`, `public/` — design notes, evaluation, training helpers, sample projects, assets.

## Build, Test, and Development Commands

The project is managed with [`uv`](https://docs.astral.sh/uv/). Always run tools via `uv run …` so the locked venv is used.

- `uv sync` — install project + dev dependencies from `pyproject.toml` / `uv.lock`.
- `uv sync --extra mnn` — also install the optional MNN embedder (~10× faster CPU inference).
- `uv run infinidev` — launch the default TUI.
- `uv run infinidev --classic` — launch the text-only renderer.
- `./install.sh` — install the wrapper for local system use (`uv tool install --force --reinstall`).
- `uv run pytest` — run the configured test suite in `tests/`.
- `uv run pytest tests/test_config.py -v` — run a single test file, verbose.
- `uv run pytest tests/test_config.py::TestConfig::test_foo -v` — run a single test by node id.
- `uv run pytest -k "loop and not slow" -v` — keyword filter, useful while iterating.
- `uv run pytest -x --tb=short` — stop on first failure with short tracebacks during debugging.

No separate lint/format/typecheck tool is configured. There is no `ruff`, `black`, `mypy`, or `isort` config in the repo — follow the style rules below and rely on the test suite.

## Coding Style & Naming Conventions

- **Indentation & layout**: idiomatic Python, 4-space indent, line length ≤ 100 chars is the working norm, `from __future__ import annotations` is used at the top of new modules.
- **Type hints**: annotate function signatures and module-level constants where they clarify contracts. Prefer modern syntax (`list[X]`, `X | None`) and `from __future__ import annotations` for forward references.
- **Naming**: `snake_case` for files/functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants and settings. Tool classes end in `Tool` (e.g. `ReadFileTool`, `EditSymbolTool`); their Pydantic input models end in `Input`; the registry entry is the `*Tool` shim.
- **Module placement**: put tool implementations in the matching `src/infinidev/tools/<category>/` package, prompt fragments under `src/infinidev/prompts/<flow_or_phase>/`, tests next to related coverage in `tests/`. Prefer existing helpers (settings, db retry, context, permissions) over new abstractions.
- **Imports**: standard library first, third-party second, local `infinidev.*` third, each group alphabetised and separated by a blank line. Use `if TYPE_CHECKING:` blocks for type-only imports that would otherwise cause cycles (see `tools/base/base_tool.py`).
- **Comments**: only for non-obvious control flow, protocol constraints, or "why" notes that are not visible from the code. Do not narrate the obvious.
- **Docstrings**: short module-level docstring on every non-trivial module; Google/NumPy-style summaries on public classes and functions.

## Error Handling & Logging

- Use module-level `logger = logging.getLogger(__name__)`; never print to stdout/stderr from library code.
- Catch the narrowest exception class that applies; let unexpected exceptions propagate so the engine's `_best_effort` / error widgets can surface them.
- DB access goes through `infinidev.tools.base.db.execute_with_retry()` with exponential backoff for SQLite WAL contention — do not open raw `sqlite3.connect()` in feature code.
- Tool runtime validates arguments via Pydantic models; return `str` (or `ToolResult` for multimodal output) from `_run`, never raise raw exceptions out of a tool unless the schema rejects the call first.
- Permission gating goes through `tools/base/permissions.py` and the `SANDBOX_ENABLED` / `ALLOWED_BASE_DIRS` settings — do not re-implement checks ad hoc.

## Configuration

Settings live in `src/infinidev/config/settings.py` (a `pydantic_settings.BaseSettings`). All env vars use the `INFINIDEV_` prefix. Project-local runtime state is created in `.infinidev/` (cwd-relative); never commit secrets, generated databases, logs, model outputs, or large finetune artifacts.

## Model Target & Context Strategy

The primary product target is one configured SOTA reasoning or coding model with a
long context window, including approximately 1M tokens when the provider and model
support it. Local open-weight models through Ollama remain supported, but they are a
compatibility path rather than the constraint that should determine the architecture.

Treat plan-execute-summarize, working memory, and recall as relevance and continuity
mechanisms, not only as token-saving workarounds. Do not justify a product-wide design
solely by 7B-model or small-window pressure. Keep existing capability probes and compact
fallbacks functional so local models do not break.

## Testing Guidelines

The suite uses `pytest` with `pytest-asyncio` in auto mode (`asyncio_mode = "auto"` in `pyproject.toml`). `pytest.ini_options` excludes `tests/finetune` and `tests/examples/taskqueue` from collection.

- Test files are `test_*.py`; functions are `test_*`. Place tests next to related coverage.
- For engine, tool, parser, or UI behavior, add a focused regression test before broader integration checks.
- Reuse the shared fixtures in `tests/conftest.py`: `temp_db`, `workspace_dir`, `tool_context`, `bound_tool`, `sandbox_disabled`/`sandbox_enabled`, `auto_approve_permissions`. The autouse fixtures already reset the SQLite cache, tool-context thread-locals, and model-capabilities singletons between tests — do not duplicate that logic.
- Iterate with `uv run pytest tests/path.py::test_name -v`, then run the full `uv run pytest` before submitting.
- Interactive tests: opt in via `INFINIDEV_RUN_INTERACTIVE_TESTS=1 uv run pytest tests/interactive`.

## Commit & Pull Request Guidelines

Recent history uses short imperative or release-style subjects, for example `hooks: wire on_file_change...`, `Fix two issues...`, and `Release 0.6.0: …`. Keep commits focused and describe the user-visible behavior or subsystem changed. Releases are tagged in-tree (version in `pyproject.toml`, e.g. `0.12.3`) — bump the version for any release commit.

Pull requests should include a concise summary, the test commands you ran, linked issues when applicable, and screenshots or terminal output for TUI-visible changes. Do not commit secrets, generated `.infinidev/` data, or large finetune checkpoints — all of these are `.gitignore`d already.

## Architecture Pointers

For deeper context than this file provides, see `CLAUDE.md` (pipeline, loop engine, tools, embeddings, key constraints) and the per-subsystem docs in `docs/`. The `ken_ken_*` tools provide indexed file/symbol/call-graph navigation when exploring the codebase.


## Code intelligence: ken

**A `<context-rank>` block in the prompt is ken's ranked guess for this
request**: `Files:` best first, then `Symbols:`, then `Notes:` — finding
*topics*, which `ken_recall(topic="…")` reads. If it names what you need,
open that and skip searching. If a listed file turns out to be irrelevant,
`ken_remember(path, action="dismiss", reason=…)` — the ranker's only
negative signal, and only while the block is still in front of you. Thin or
missing? `ken_rank(verbose=2)`, or `ken_find(task, scope="intent")` for the
files that work like this one actually landed in.

**Start with one `ken_find`, not with `rg` or a guessed path.** The scope is
the whole decision:

- an exact string or identifier (`MY_ENV_VAR`, `os.path`) → `scope="text", literal=true`
- which file does X → `scope="files"`
- which function or class does X → `scope="symbols"`
- how a route, CLI command or env var reaches its handler → `scope="wiring"`

Then read what it named — `ken_read(path)` for the outline, plus
`include=["source"]` and a *qualname* for one symbol's body. ken narrows
where to look; it does not replace reading the code.

**Stop rules.** If two ken calls have not narrowed it, open the likeliest
file and read — a third will not help. If ken returns nothing, use `rg`: it
searches the index, so a file created minutes ago may not be in it yet. A
question you can already answer from context needs no call at all.

**Before editing a file you have not read this session**, `ken_recall(path=…)`
— what earlier sessions learned there. If the change is not local:
`ken_related(path, relation="blast_radius")` for what it breaks,
`relation="cochange"` for what moves with it that imports do not show, and
`ken_find(path, scope="tests")` for its tests.

**Write back what cost you real effort** — a root cause, a constraint the
code does not state, a trap you fell into: `ken_remember(topic, content)`.
Not what the code already says plainly. Re-using a topic overwrites it.

**Anchor it, or it only fires if someone searches.** An anchored memory is
handed to whoever next touches the same thing — no query needed:
`ken_remember(topic, content, anchor_file="src/db/service.py")`, or
`anchor_symbol` for one function, `anchor_tool="pytest"` for a command,
`anchor_error="database is locked"` for a message (matched as a substring of
whatever the tool reports). Set as many as apply; the memory fires on any.
