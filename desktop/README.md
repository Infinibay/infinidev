# Infinidev Desktop

A cross-platform desktop GUI for Infinidev — a **Tauri 2** native shell, a
**React 19 + Vite + Harbor UI** frontend, and the agent engine **rewritten in
Rust and embedded in-process** (no Python sidecar, no local HTTP server).

```
┌─────────────────────────────────────────────────────────────┐
│  Tauri 2 (Rust)                                              │
│   • embeds the Rust agent core (crates/infinidev-*)           │
│   • commands: engine_send, set_config, list_provider_models,  │
│     fs_tree/read/write, project_search, …                     │
│   • streams EngineEvents to the WebView as `engine://event`   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  WebView — React 19 + Harbor UI                          │ │
│  │   conversation-first workbench: chat + live tool cards,   │ │
│  │   Activity / Changes / Files / Terminal, model + cost     │ │
│  │   talks to the engine via Tauri invoke + listen           │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

The Rust core lives in `../crates/`:

| Crate | Role |
| --- | --- |
| `infinidev-llm` | provider registry (OpenAI/Anthropic/Ollama/…), chat client (+ SSE streaming), capabilities, **pricing** |
| `infinidev-tools` | file / search / shell / git tools as a `Tool` trait |
| `infinidev-knowledge` | SQLite findings store (record/search/read) |
| `infinidev-engine` | the tool-calling loop + `EngineEvent` stream |

The frontend is transport-agnostic: in the desktop build it uses the Tauri
transport (real Rust engine); in a plain browser (`npm run dev`) it uses a
**demo transport** that replays a scripted turn, so the UI is fully exercisable
without a backend.

> The Python package (`../src/infinidev`) remains the Textual **TUI** and the
> reference implementation. The desktop GUI no longer depends on it.

## Prerequisites

- **Node.js ≥ 20** and npm
- **Rust** (stable) via [rustup](https://rustup.rs) + your platform's
  [Tauri 2 system dependencies](https://v2.tauri.app/start/prerequisites/)
- **Ollama** (or any OpenAI-compatible provider) running locally with a model
- `git` — Harbor is a submodule, so clone with `--recursive`

## Setup

```bash
git submodule update --init --recursive   # vendor Harbor UI
cd desktop && npm install
```

## Run

```bash
# Desktop (native shell + embedded Rust engine)
cd desktop && npm run tauri:dev

# UI only, in a browser (demo transport — no engine)
cd desktop && npm run dev          # http://localhost:5173
```

By default the engine operates on the current working directory; set
`INFINIDEV_PROJECT=/path/to/project` to point it elsewhere. Pick the model and
provider from the in-app settings (the ⚙ / model chip in the top bar).

## Checks

```bash
npm run typecheck                       # frontend (tsc -b --noEmit)
npm run build                           # production web build → dist/
(cd .. && cargo test)                   # Rust core (llm/tools/engine/knowledge)
(cd src-tauri && cargo check)           # native shell
```

## Packaging

```bash
cd desktop && npm run tauri:build
```

Because the engine is native Rust compiled into the app, there is **no Python
runtime to bundle** — `tauri build` produces a single self-contained binary per
platform (cross-platform binaries must be built on each target OS).

## License

Harbor UI is **AGPL-3.0-or-later**; distributing this app means complying with
AGPL's source-offer obligations for the bundled Harbor code.

## Layout

```
desktop/
├── index.html · vite.config.ts · tailwind.config.js   # Vite + Harbor token bridge
├── external/infinibay_ui/        # Harbor UI (git submodule — do not edit)
├── src/
│   ├── platform/tauri.ts         # invoke + engine://event bridge
│   ├── api/{engineEvent,transport,tauri,demo}.ts   # transports + commands
│   ├── state/store.tsx           # EngineEvent → UI state
│   └── ui/                       # Topbar, conversation/, workbench/, modals/
└── src-tauri/                    # Rust shell embedding ../../crates/infinidev-engine
```
