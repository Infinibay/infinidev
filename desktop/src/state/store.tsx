/**
 * Application store. The engine (Rust, embedded in Tauri) streams
 * `EngineEvent`s; this reducer folds them into the UI state the
 * conversation-first workbench renders. In a browser it transparently uses the
 * demo transport so the UI is fully exercisable without a backend.
 */
import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  type ReactNode,
} from "react";
import { isTauri } from "@/platform/tauri";
import { tauriCommands, tauriTransport } from "@/api/tauri";
import { demoCommands, demoTransport } from "@/api/demo";
import type { Commands, Transport } from "@/api/transport";
import type { AskPrompt, EngineEvent } from "@/api/engineEvent";

export type ChatKind = "user" | "agent" | "tool" | "system" | "error";

export interface ChatMessage {
  id: string;
  kind: ChatKind;
  text: string;
  streaming?: boolean;
  /** Attached images (`data:` URLs) on a user message. */
  images?: string[];
  // tool messages
  toolId?: string;
  toolName?: string;
  toolArgs?: unknown;
  toolOutput?: string;
  toolOk?: boolean;
  toolDone?: boolean;
}

export interface ChangedFile {
  path: string;
  action: string;
}

export interface Decision {
  kind: "respond" | "escalate";
  detail: string;
}

export interface Review {
  status: "approve" | "changes";
  notes: string[];
}

export interface AppState {
  ready: boolean;
  transport: "tauri" | "demo";
  projectPath: string;
  /** Bumped whenever the project folder changes, to retrigger folder-scoped loads. */
  projectNonce: number;
  model: string;
  busy: boolean;
  phase: string;
  step: { n: number; max: number } | null;
  plan: string[];
  messages: ChatMessage[];
  reasoning: string;
  changedFiles: ChangedFile[];
  decision: Decision | null;
  review: Review | null;
  usage: { prompt: number; completion: number; total: number } | null;
  costUsd: number;
  /** A host question the engine is blocking on (command confirmation, etc.). */
  ask: AskPrompt | null;
}

function initial(): AppState {
  return {
    ready: false,
    transport: "demo",
    projectPath: "",
    projectNonce: 0,
    model: "",
    busy: false,
    phase: "idle",
    step: null,
    plan: [],
    messages: [],
    reasoning: "",
    changedFiles: [],
    decision: null,
    review: null,
    usage: null,
    costUsd: 0,
    ask: null,
  };
}

type Action =
  | { kind: "engineEvent"; ev: EngineEvent }
  | { kind: "localUser"; text: string; images?: string[] }
  | { kind: "init"; projectPath: string; model: string; transport: "tauri" | "demo" }
  | { kind: "setModel"; model: string }
  | { kind: "setProjectPath"; path: string }
  | { kind: "setAsk"; ask: AskPrompt | null }
  | { kind: "restore"; history: { role: string; content: string }[] }
  | { kind: "clear" };

let seq = 0;
const id = () => `m${++seq}`;

function push(s: AppState, m: Omit<ChatMessage, "id">): AppState {
  return { ...s, messages: [...s.messages, { id: id(), ...m }] };
}

function appendStream(s: AppState, chunk: string): AppState {
  const last = s.messages[s.messages.length - 1];
  if (last && last.kind === "agent" && last.streaming) {
    const updated = { ...last, text: last.text + chunk };
    return { ...s, messages: [...s.messages.slice(0, -1), updated] };
  }
  return push(s, { kind: "agent", text: chunk, streaming: true });
}

function finalizeStream(s: AppState): AppState {
  const last = s.messages[s.messages.length - 1];
  if (last && last.streaming) {
    return { ...s, messages: [...s.messages.slice(0, -1), { ...last, streaming: false }] };
  }
  return s;
}

function updateTool(s: AppState, toolId: string, ok: boolean, output: string): AppState {
  return {
    ...s,
    messages: s.messages.map((m) =>
      m.toolId === toolId ? { ...m, toolOk: ok, toolOutput: output, toolDone: true } : m,
    ),
  };
}

function upsertFile(files: ChangedFile[], path: string, action: string): ChangedFile[] {
  const idx = files.findIndex((f) => f.path === path);
  if (idx >= 0) {
    const copy = files.slice();
    copy[idx] = { path, action: files[idx].action === "create" ? "create" : action };
    return copy;
  }
  return [...files, { path, action }];
}

function applyEvent(s: AppState, ev: EngineEvent): AppState {
  switch (ev.type) {
    case "phase":
      return { ...s, phase: ev.phase, busy: ev.phase !== "idle" };
    case "decision":
      return { ...s, decision: { kind: ev.kind, detail: ev.detail } };
    case "plan":
      return { ...s, plan: ev.steps };
    case "step_start":
      return { ...s, step: { n: ev.step, max: ev.max }, busy: true };
    case "stream_chunk":
      return appendStream(s, ev.chunk);
    case "reasoning":
      return { ...s, reasoning: s.reasoning + ev.chunk };
    case "notify": {
      const kind: ChatKind = ev.kind === "error" ? "error" : ev.kind === "system" ? "system" : "agent";
      return push(finalizeStream(s), { kind, text: ev.text });
    }
    case "tool_call":
      return push(s, {
        kind: "tool",
        text: "",
        toolId: ev.id,
        toolName: ev.name,
        toolArgs: ev.args,
        toolDone: false,
      });
    case "tool_result":
      return updateTool(s, ev.id, ev.ok, ev.output);
    case "file_change":
      return { ...s, changedFiles: upsertFile(s.changedFiles, ev.path, ev.action) };
    case "usage":
      return {
        ...s,
        usage: { prompt: ev.prompt_tokens, completion: ev.completion_tokens, total: ev.total_tokens },
        costUsd: s.costUsd + (ev.cost_usd ?? 0),
      };
    case "review":
      return { ...s, review: { status: ev.status, notes: ev.notes } };
    case "turn_end":
      return { ...finalizeStream(s), busy: false, step: null, phase: "idle", ask: null };
    case "error":
      return push(s, { kind: "error", text: ev.message });
    default:
      return s;
  }
}

function reducer(s: AppState, a: Action): AppState {
  switch (a.kind) {
    case "engineEvent":
      return applyEvent(s, a.ev);
    case "localUser":
      return {
        ...push(s, { kind: "user", text: a.text, images: a.images?.length ? a.images : undefined }),
        busy: true,
        reasoning: "",
        phase: "working",
        plan: [],
        decision: null,
        review: null,
      };
    case "init":
      return { ...s, ready: true, projectPath: a.projectPath, model: a.model, transport: a.transport };
    case "setModel":
      return { ...s, model: a.model };
    case "setProjectPath":
      return { ...s, projectPath: a.path, projectNonce: s.projectNonce + 1 };
    case "setAsk":
      return { ...s, ask: a.ask };
    case "restore": {
      // Redraw a persisted conversation on startup (user/assistant turns only).
      const messages: ChatMessage[] = a.history
        .map((m): ChatMessage | null => {
          if (m.role === "user") return { id: id(), kind: "user", text: m.content };
          if (m.role === "assistant") return { id: id(), kind: "agent", text: m.content };
          return null;
        })
        .filter((m): m is ChatMessage => m !== null);
      return { ...s, messages };
    }
    case "clear":
      return { ...initial(), ready: s.ready, transport: s.transport, projectPath: s.projectPath, model: s.model };
  }
}

export interface EngineApi {
  state: AppState;
  commands: Commands;
  transportKind: "tauri" | "demo";
  send: (text: string, images?: string[]) => void;
  reset: () => void;
  /** Ask the engine to abort the turn in flight (no-op when idle). */
  stop: () => void;
  /** Answer the pending host question (command confirmation, etc.). */
  answerPrompt: (answer: string) => void;
  setModel: (model: string) => void;
  /** Re-read the project path from the backend and retrigger folder-scoped loads. */
  refreshProject: () => Promise<void>;
}

const Ctx = createContext<EngineApi | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, undefined, initial);
  const ref = useRef<{ transport: Transport; commands: Commands }>(null!);
  if (!ref.current) {
    const t = isTauri();
    ref.current = {
      transport: t ? tauriTransport : demoTransport,
      commands: t ? tauriCommands : demoCommands,
    };
  }

  useEffect(() => {
    let alive = true;
    let unlisten = () => {};
    let unlistenAsk = () => {};
    (async () => {
      const un = await ref.current.transport.subscribe((ev) => dispatch({ kind: "engineEvent", ev }));
      const unAsk = await ref.current.transport.subscribeAsk((ask) => dispatch({ kind: "setAsk", ask }));
      if (alive) {
        unlisten = un;
        unlistenAsk = unAsk;
      } else {
        un();
        unAsk();
      }
      try {
        const [pp, cfg, history] = await Promise.all([
          ref.current.commands.projectPath(),
          ref.current.commands.getConfig(),
          ref.current.commands.getHistory().catch(() => []),
        ]);
        if (alive) {
          dispatch({ kind: "init", projectPath: pp, model: cfg.model, transport: ref.current.transport.kind });
          if (history.length) dispatch({ kind: "restore", history });
        }
      } catch {
        if (alive) dispatch({ kind: "init", projectPath: "", model: "", transport: ref.current.transport.kind });
      }
    })();
    return () => {
      alive = false;
      unlisten();
      unlistenAsk();
    };
  }, []);

  const api = useMemo<EngineApi>(
    () => ({
      state,
      commands: ref.current.commands,
      transportKind: ref.current.transport.kind,
      send: (text: string, images?: string[]) => {
        const t = text.trim();
        if ((!t && !images?.length) || state.busy) return;
        dispatch({ kind: "localUser", text: t, images });
        void ref.current.transport.send(t, images);
      },
      reset: () => {
        void ref.current.transport.reset();
        dispatch({ kind: "clear" });
      },
      stop: () => {
        if (!state.busy) return;
        void ref.current.transport.cancel();
      },
      answerPrompt: (answer: string) => {
        const a = state.ask;
        if (!a) return;
        void ref.current.transport.answer(a.id, answer);
        dispatch({ kind: "setAsk", ask: null });
      },
      setModel: (model: string) => dispatch({ kind: "setModel", model }),
      refreshProject: async () => {
        try {
          const pp = await ref.current.commands.projectPath();
          dispatch({ kind: "setProjectPath", path: pp });
        } catch {
          /* leave the current path on failure */
        }
      },
    }),
    [state],
  );

  return <Ctx.Provider value={api}>{children}</Ctx.Provider>;
}

export function useEngine(): EngineApi {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useEngine must be used within <AppProvider>");
  return ctx;
}

export function useAppState(): AppState {
  return useEngine().state;
}
