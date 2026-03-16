# Infinidev

A terminal-based AI programming assistant powered by local LLMs. It runs an autonomous agent loop that can read, write, and edit code, execute commands, manage git, search the web, and maintain a persistent knowledge base — all from your terminal.

Designed to work with open-weight models (7B-14B) running on consumer hardware via [Ollama](https://ollama.com).

![Infinidev TUI](public/screenshot.png)

## Features

- **Plan-execute-summarize loop** — the agent breaks tasks into steps, executes them with tools, and summarizes results. No bloated context windows.
- **Full-featured TUI** — tabbed interface with chat, file explorer, syntax-highlighted editor, sidebar with live progress, and autocomplete for commands.
- **Persistent knowledge base** — the agent records what it learns about your project (classes, patterns, APIs) and recalls it in future sessions. Critical for small models.
- **Dual tool-calling modes** — auto-detects whether the LLM supports native function calling or falls back to JSON-in-text parsing.
- **20+ built-in tools** — file operations, git, shell, web search/fetch, knowledge management with semantic dedup.
- **Project-local state** — settings, DB, and logs live in `.infinidev/` inside your project directory.
- **Model management** — list, switch, and interactively pick Ollama models from the TUI.

## Requirements

- Python 3.13+
- [Ollama](https://ollama.com) running locally (or any LiteLLM-compatible provider)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Quickstart

```bash
# Clone and install
git clone https://github.com/yourusername/infinidev.git
cd infinidev
uv sync

# Make sure Ollama is running with a model
ollama pull qwen2.5-coder:7b

# Launch
uv run infinidev
```

Or install system-wide:

```bash
./install.sh
infinidev
```

## Usage

### TUI Mode (default)

```bash
uv run infinidev
```

The TUI has three panels:
- **Left** — File explorer (toggle with `Ctrl+E`)
- **Center** — Tabbed area with Chat + file editor tabs
- **Right** — Sidebar showing plan progress, active tools, and logs

### Classic Mode

```bash
uv run infinidev --no-tui
```

Text-only mode for minimal terminals or piping.

### Commands

| Command | Description |
|---------|-------------|
| `/help` | Show all commands and keybindings |
| `/models` | Show current model configuration |
| `/models list` | List available Ollama models |
| `/models set <name>` | Change the active model |
| `/models manage` | Interactive model picker |
| `/findings` | Browse all knowledge base findings |
| `/knowledge` | Browse project context knowledge |
| `/clear` | Clear chat history and panels |
| `/exit` | Quit |

### Keybindings

| Key | Action |
|-----|--------|
| `Ctrl+E` | Toggle file explorer |
| `Ctrl+W` | Close active file tab |
| `F2` / `F3` / `F4` | Focus: Chat / Explorer / Sidebar |

## Configuration

Settings are stored in `.infinidev/settings.json` in your project directory. They can also be set via environment variables with the `INFINIDEV_` prefix.

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODEL` | `ollama_chat/qwen2.5-coder:7b` | LiteLLM model identifier |
| `LLM_BASE_URL` | `http://localhost:11434` | Ollama / LLM API base URL |
| `LOOP_MAX_ITERATIONS` | `50` | Max planning iterations per task |
| `LOOP_MAX_TOTAL_TOOL_CALLS` | `200` | Global tool call limit per task |
| `LOOP_HISTORY_WINDOW` | `0` | Summaries to keep (0 = all) |

## Architecture

```
src/infinidev/
  cli/          # TUI (Textual) and classic CLI entry points
  engine/       # Plan-execute-summarize loop engine
  agents/       # Agent role definitions and tool binding
  tools/        # 20+ tools: file, git, shell, web, knowledge
  config/       # Settings, LLM params, model capability probing
  db/           # SQLite with FTS5, findings, artifacts, conversations
  prompts/      # System prompts, tech-specific guidelines
```

The core loop:
1. **Plan** — LLM produces 2-3 initial steps
2. **Execute** — one step at a time, calling tools as needed
3. **Summarize** — LLM produces a compact summary; raw output is discarded
4. **Repeat** — prompt is rebuilt from scratch each iteration using only summaries

This keeps the context window small and predictable, which is critical for 7B models.

## Knowledge Base

The agent maintains a persistent knowledge base of findings across sessions. It automatically records:
- Project structure, key classes, and public APIs
- Patterns, conventions, and dependencies
- Research results and bug findings
- Things you ask it to remember

Findings are auto-injected into the prompt at the start of each task, so the agent starts every session already knowing your project.

Browse the knowledge base anytime with `/findings` or `/knowledge`.

## Development

```bash
# Run tests
uv run pytest tests/

# Run a specific test
uv run pytest tests/test_tui.py::test_space_inserts_space_character -v
```
