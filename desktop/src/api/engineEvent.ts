/**
 * The Rust engine's event stream (mirrors `infinidev_engine::EngineEvent`,
 * serialized with a snake_case `type` tag). Delivered over the Tauri
 * `engine://event` channel and fed into the store.
 */
export type EngineEvent =
  | { type: "phase"; phase: string }
  | { type: "decision"; kind: "respond" | "escalate"; detail: string }
  | { type: "plan"; steps: string[] }
  | { type: "step_start"; step: number; max: number }
  | { type: "stream_chunk"; chunk: string }
  | { type: "reasoning"; chunk: string }
  | { type: "notify"; speaker: string; text: string; kind: string }
  | { type: "tool_call"; id: string; name: string; args: unknown }
  | { type: "tool_result"; id: string; name: string; ok: boolean; output: string }
  | { type: "file_change"; path: string; action: string }
  | {
      type: "usage";
      prompt_tokens: number;
      completion_tokens: number;
      total_tokens: number;
      cost_usd: number | null;
    }
  | { type: "review"; status: "approve" | "changes"; notes: string[] }
  | { type: "turn_end"; result: string }
  | { type: "error"; message: string };

export interface EngineConfig {
  provider: string;
  model: string;
  api_key: string | null;
  base_url: string;
  temperature: number | null;
  max_iterations: number;
  planning: boolean;
  orchestrate: boolean;
  review: boolean;
  force_manual_tools: boolean;
  confirm_commands: boolean;
  confirm_writes: boolean;
}

/** A question the engine is blocking on (e.g. confirm a shell command). */
export interface AskPrompt {
  id: number;
  prompt: string;
  /** "permission" | "input" | … — lets the UI pick allow/deny vs a text field. */
  kind: string;
}

export interface ProviderDto {
  id: string;
  display_name: string;
  api_key_required: boolean;
  base_url_editable: boolean;
  default_base_url: string;
}

export interface RsTreeEntry {
  name: string;
  path: string;
  is_dir: boolean;
}

export interface RsSearchHit {
  path: string;
  line: number;
  text: string;
}

/** A persisted finding from the project knowledge base (`infinidev-knowledge`). */
export interface Finding {
  id: number;
  topic: string;
  content: string;
  finding_type: string;
  confidence: number;
  created_at: string;
}

/** A working-memory note the agent recorded via `record_note`. */
export interface NoteRow {
  id: number;
  note: string;
  created_at: string;
}

/** Live view of a background task (a command started with `run_in_background`). */
export interface BgTask {
  id: number;
  command: string;
  description: string;
  status: string;
  running: boolean;
  exit_code: number | null;
  output_tail: string;
}

/** A library with cached documentation. */
export interface DocLibrary {
  library: string;
  version: string;
  sections: number;
}

/** A cached documentation section header. */
export interface DocSection {
  section_title: string;
  section_order: number;
  source_url: string;
}

/** A search hit across cached docs. */
export interface DocHit {
  library: string;
  section_title: string;
  snippet: string;
}
