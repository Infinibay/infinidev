# Infinidev Web

## Product direction

The web application is a workspace for operating the existing harness. A user gives
the principal agent a normal request; the interface makes its execution, collaborators,
evidence, and pending decisions visible. It does not introduce a separate agent engine.

The first deployment is one local server for one project and user. Sessions survive
browser refreshes. A browser connection observes a session; disconnecting it must not
silently approve a permission request, cancel a task, or start a duplicate run.

## Information architecture

| Area | User work |
| --- | --- |
| Workspace | Send requests, steer an active run, answer questions, and stop execution |
| Sessions | Find, resume, and rename conversations with their actual execution status |
| Team | See named specialists, responsibilities, tool grants, tickets, dependencies, and results |
| Shared notes | Read observations, hypotheses, decisions, and handoffs with authors and references |
| Processes | Inspect background commands, open live stdout/stderr, and stop a selected process |
| Changes | Review the working tree, inspect diffs, and open affected source files |
| Knowledge | Search saved findings and inspect their supporting evidence |
| Activity | Follow tool calls, progress, errors, and model-reported reasoning when available |
| Configuration | Choose providers, models, and supported effort; inspect tools and runtime settings |
| Usage | Read observed consumption and the selected provider's available quota or billing report |

Conversation is the default view. Teams and processes have dedicated views and can also
be opened beside a conversation. Questions and permission requests remain prominent until
answered; approving an action requires an explicit user response. Tool details, reasoning,
and raw evidence use progressive disclosure so they do not bury the agent's answer.

## Runtime contracts

- Reuse `run_task`, orchestration hooks, cooperative cancellation, the session database,
  team store, background manager, permission handler, model registry, and effort profiles.
- Keep network I/O and browser updates off the synchronous engine worker. Serialize
  independent session turns while the existing runtime shares process-level settings,
  event subscriptions, and a writable workspace.
- Persist conversation history and use ordered event IDs for browser reconnects. A
  reconnect returns the current state and pending questions; it never resubmits a prompt.
- Separate persisted team state from worker liveness. Keep agent IDs for routing and
  attribution while displaying the orchestrator's human names and descriptive roles.
- Use bounded output buffers and explicit unavailable/error states. Never fill an empty
  production workspace with invented agents, metrics, findings, or successful actions.
- Serve the React build locally. Model keys remain on the Python side, settings responses
  redact secrets, and browser access does not bypass existing filesystem permissions.

## Visual direction

A quiet, dense workspace: graphite navigation, warm neutral surfaces, restrained green
status accents, readable typography, and clear separators. Avoid a dashboard full of
decorative counters. Keep the conversation comfortable to read while making a team's
ownership and progress legible at a glance. Support keyboard navigation, narrow screens,
empty states, reconnecting states, and long-running work.

## Beyond the initial deployment

Remote multi-user access requires authenticated accounts, workspace isolation in separate
runtime processes, and authorization for each resource. Experiment comparison, reproducible
run bundles, richer artifact previews, and per-agent model overrides are natural additions;
they should build on recorded evidence and explicit runtime capabilities.

## Run from this checkout

```bash
uv sync --locked --extra web
cd web
npm ci
npm run build
cd ..
uv run --locked --extra web infinidev web --workdir /path/to/project
```

`infinidev serve` is an alias. `--no-open` prints the authenticated launch URL
without opening a browser; `--port` defaults to 8765. The normal `infinidev`
command still launches the terminal interface. Packaged releases include the web
build; install their optional server dependencies with `infinidev[web]`.

The first version binds only to loopback. The launch URL contains a random access
capability in its fragment, which is removed from the address bar after loading.
The browser keeps it in session storage. REST calls use a bearer header; WebSocket
handshakes carry it in a subprotocol. Foreign origins and hosts are rejected.
Model credentials stay in the existing project settings and are never returned by
the settings API. Authentication does not change the engine's permission policy.

For frontend development, keep the Python server running on port 8765 and run
`npm run dev` in `web/`. Open the Vite URL with the same `#token=…` fragment from
the server's launch URL. Vite proxies `/api` and `/ws` to Python.

## Included workflows and limits

- Send a normal request; the configured default orchestrator uses the existing
  prompt assembly, repository rules, tools and research capabilities. There is no
  web-specific system prompt replacing those rules.
- Resume or rename a project session. Streaming answers, tool activity, questions
  and permissions survive browser reconnections. In-flight execution stays in the
  server process; stopping the server does not promise execution recovery.
- Inspect specialist names, roles, exact prompts and granted tools; browse ticket
  objectives, acceptance conditions, results and reviews. Shared notes and team
  messages retain author labels. User-authored notes are stored as `user`.
- Open background processes in output tabs, follow retained stdout/stderr, copy or
  download the retained log, and stop the process. Output refreshes every 750 ms.
  The manager belongs to this server process; processes in another TUI instance
  are not represented as live web processes.
- Review staged and unstaged diffs. Browse and edit UTF-8 project files up to 2 MB;
  saves check the revision read by the browser and respect file permission checks.
  An external edit requires reloading before overwriting. Git commit/push actions
  remain available through the conversation and existing agent tools.
- Search the project's Infinidev findings and tool catalog, including discovered
  MCP tools. Ken's separate database remains accessible through its agent tools;
  the Findings page does not claim to aggregate that external database.
- Configure providers, models and model-specific effort; discover live model IDs.
  Provider switching clears the previous provider's key unless explicitly replaced.
  Subscription providers reuse the existing CLI login. Usage invokes the same
  provider-aware report as `/usage`.
- Use the command palette with Cmd/Ctrl+K, switch themes, and use the workspace on
  narrow screens. Conversation drafts are retained while switching sessions/views.

Main-agent turns are serialized across sessions because settings, hooks and the
workspace are process-global. Specialist concurrency remains controlled by the
orchestrator. The browser retains the latest 500 transcript entries and 1,000 team
events; older persisted messages remain in the session database. Team state is
explicitly marked inactive when no live team runtime is attached. Image uploads,
a dedicated experiment comparison UI and remote multi-user operation are outside
this first version.

## Validation and packaging

```bash
uv run --locked --extra web pytest
cd web
npm test
npm run build
npx playwright install chromium
npm run test:e2e
```

The browser tests start an isolated temporary project with deterministic execution
fixtures. They exercise the real HTTP/WebSocket server without calling paid models.
Fixture agents, findings and processes are confined to `web/e2e/server.py`; production
never seeds demonstration data. Browser screenshots and traces go to ignored
`web/test-results/`.

CI builds the frontend before creating release distributions. Hatch includes the
built static files in the wheel and source distribution, alongside the TypeScript
source in the latter. Radix UI is MIT licensed; the UI does not copy Harbor code.
`npm run build` generates `THIRD-PARTY-NOTICES.txt` with production dependency and
font license notices, shipped beside the static application.
