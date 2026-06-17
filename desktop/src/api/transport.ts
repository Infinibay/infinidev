/**
 * Transport abstraction. The desktop build talks to the embedded Rust engine
 * over Tauri (`invoke` + `engine://event`); a browser build (preview/dev) uses
 * a scripted demo transport so the whole UI is exercisable without a backend.
 */
import type {
  AskPrompt,
  BgTask,
  DocHit,
  DocLibrary,
  DocSection,
  EngineConfig,
  EngineEvent,
  Finding,
  NoteRow,
  ProviderDto,
  RsSearchHit,
  RsTreeEntry,
} from "./engineEvent";

export interface ConfigPatch {
  provider?: string;
  model?: string;
  api_key?: string | null;
  base_url?: string;
  temperature?: number;
  max_iterations?: number;
  planning?: boolean;
  orchestrate?: boolean;
  review?: boolean;
  force_manual_tools?: boolean;
  confirm_commands?: boolean;
  confirm_writes?: boolean;
}

/** The live event channel + turn control. */
export interface Transport {
  kind: "tauri" | "demo";
  subscribe(cb: (ev: EngineEvent) => void): Promise<() => void>;
  /** Run a turn. `images` are `data:` URLs attached to the user message. */
  send(text: string, images?: string[]): Promise<void>;
  reset(): Promise<void>;
  /** Abort the turn currently running (best-effort; no-op when idle). */
  cancel(): Promise<void>;
  /** Subscribe to host questions the engine blocks on (e.g. command confirmation). */
  subscribeAsk(cb: (p: AskPrompt) => void): Promise<() => void>;
  /** Answer a pending host question by id. */
  answer(id: number, answer: string): Promise<void>;
}

/** Request/response commands for non-conversational data. */
export interface Commands {
  projectPath(): Promise<string>;
  /** Switch the active project folder; returns the resolved absolute path. */
  setProject(path: string): Promise<string>;
  getConfig(): Promise<EngineConfig>;
  setConfig(patch: ConfigPatch): Promise<EngineConfig>;
  providers(): Promise<ProviderDto[]>;
  listModels(provider: string, apiKey: string | null, baseUrl: string): Promise<string[]>;
  fsTree(path: string): Promise<RsTreeEntry[]>;
  fsRead(path: string): Promise<{ path: string; text: string; binary: boolean }>;
  fsWrite(path: string, text: string): Promise<void>;
  /** Create an empty file or a directory. */
  fsCreate(path: string, dir: boolean): Promise<void>;
  /** Move/rename a file or directory. */
  fsRename(from: string, to: string): Promise<void>;
  /** Delete a file, or a directory and its contents. */
  fsDelete(path: string): Promise<void>;
  search(query: string): Promise<RsSearchHit[]>;
  /** Project knowledge-base findings; returns the recent list when `query` is blank. */
  findings(query: string): Promise<Finding[]>;
  /** Record a new finding; returns its id. */
  recordFinding(topic: string, content: string, findingType: string): Promise<number>;
  /** Unified diff of a file vs git HEAD (empty string when unchanged). */
  gitDiff(path: string): Promise<string>;
  /** Run a shell command (user-initiated, from the terminal) in the project dir. */
  runCommand(command: string): Promise<CommandResult>;
  /** Prior conversation (user/assistant turns) to redraw on startup. */
  getHistory(): Promise<HistoryMsg[]>;
  /** The agent's working-memory notes (oldest first), for the Notes browser. */
  notesList(): Promise<NoteRow[]>;
  /** Clear all recorded notes. */
  notesClear(): Promise<void>;
  /** Live snapshot of background tasks (dev servers, watchers, builds). */
  bgList(): Promise<BgTask[]>;
  /** Kill a background task by id; false if there's no such task. */
  bgStop(id: number): Promise<boolean>;
  /** Libraries with cached documentation. */
  docsLibraries(): Promise<DocLibrary[]>;
  /** Section headers for a library. */
  docsSections(library: string): Promise<DocSection[]>;
  /** Full content of one cached section. */
  docsRead(library: string, section: string): Promise<string>;
  /** Search across all cached docs. */
  docsSearch(query: string): Promise<DocHit[]>;
  /** Fetch a docs URL and cache it; returns the number of sections stored. */
  docsFetch(library: string, url: string, version?: string): Promise<number>;
  /** Remove a library's cached docs. */
  docsDelete(library: string): Promise<void>;
}

export interface HistoryMsg {
  role: string;
  content: string;
}

export interface CommandResult {
  stdout: string;
  stderr: string;
  code: number;
}
