# Infinidev

[![PyPI](https://img.shields.io/pypi/v/infinidev.svg)](https://pypi.org/project/infinidev/)
[![Python](https://img.shields.io/pypi/pyversions/infinidev.svg)](https://pypi.org/project/infinidev/)
[![Tests](https://github.com/Infinibay/infinidev/actions/workflows/tests.yml/badge.svg)](https://github.com/Infinibay/infinidev/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)

Infinidev is an autonomous AI coding agent for the terminal. It can inspect a
repository, plan changes, edit code, run commands and tests, review its work,
and keep project knowledge across sessions.

It works with hosted models through LiteLLM and with local models through
[Ollama](https://ollama.com).

![Infinidev TUI](https://raw.githubusercontent.com/Infinibay/infinidev/main/public/screenshot.png)

## Install

Python 3.11 or newer is required. The recommended installation uses
[uv](https://docs.astral.sh/uv/), which keeps Infinidev in an isolated tool
environment:

```bash
uv tool install infinidev
```

Upgrade later with:

```bash
uv tool upgrade infinidev
```

You can also install it with pipx:

```bash
pipx install infinidev
```

## Start

Run Infinidev from the project you want it to work on:

```bash
cd path/to/your/project
infinidev
```

Use `/models manage` to choose a provider and model, or `/settings` to edit the
project configuration. By default, Infinidev expects Ollama at
`http://localhost:11434`.

Useful command-line modes:

```bash
infinidev --no-tui                 # text-only interface
infinidev -p "explain this repo"   # one prompt, then exit
infinidev --continue               # resume the latest local session
infinidev --resume                 # choose a previous session
```

## What it includes

- A live terminal UI with task progress, diffs, files, logs, and context usage.
- Plan-execute-summarize and staged execution for long, multi-step tasks.
- File, shell, Git, web, code-intelligence, knowledge, and image tools.
- Persistent, searchable working memory with recoverable tool evidence.
- Model capability detection and native or text-based tool calling.
- MCP server support; Ken is used for semantic project context when available.
- Permission checks, deterministic completion gates, tests, and post-change review.
- Image attachments, terminal image viewing, and opt-in image generation.

## Essential commands

| Command | Purpose |
| --- | --- |
| `/help` | Show commands and keybindings |
| `/models manage` | Choose the active provider and model |
| `/settings` | View or change project settings |
| `/engine` | Select the task execution engine |
| `/plan <task>` | Review a plan before execution |
| `/mcp` | Show MCP server health |
| `/findings` | Browse saved project knowledge |
| `/reindex` | Rebuild the local code index |
| `/clear` | Clear the transcript |
| `/exit` | Quit |

## How it works

```text
request
  -> read-only chat agent
  -> planner (when code work is needed)
  -> plan-execute-summarize loop
  -> tests and objective checks
  -> review
  -> result
```

Infinidev stores settings, history, logs, and its SQLite knowledge database in
`.infinidev/` inside the current project. Environment variables use the
`INFINIDEV_` prefix. Secrets and generated runtime state should not be committed.

Council transcripts shown by `/agents` are process-local and retain the 100 most
recent completed councils by default, while always preserving active councils. Set
`COUNCIL_HISTORY_LIMIT` in `.infinidev/settings.json` (or
`INFINIDEV_COUNCIL_HISTORY_LIMIT`) to a non-negative number; use `0` to discard a
transcript after its completion event or `null` for unlimited retention.

## Development

```bash
git clone https://github.com/Infinibay/infinidev.git
cd infinidev
uv sync
uv run pytest
uv run infinidev
```

Architecture and subsystem documentation live in [`docs/`](docs/). Contributor
guidance is in [`AGENTS.md`](AGENTS.md).

## License

The Infinidev software and documentation are available under the [MIT License](LICENSE.md).
The manually authored task-policy annotations in
[`data/task-policy-reviews/`](data/task-policy-reviews/) are a separately licensed data
component; see their [data license and attribution notice](data/task-policy-reviews/LICENSE.md).
