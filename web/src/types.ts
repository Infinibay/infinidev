export type View =
  | "workspace"
  | "team"
  | "processes"
  | "changes"
  | "knowledge"
  | "tools"
  | "usage"
  | "settings";
export type RecordData = Record<string, unknown>;
export interface SessionSummary {
  session_id: string;
  title: string;
  status: string;
  phase: string;
  last_active_at?: string;
}
export interface Message {
  id: string;
  speaker: string;
  text: string;
  kind: string;
  state?: string;
  streaming?: boolean;
  created_at?: number;
  data?: RecordData;
  agent_id?: string;
  traceback?: string;
}
export interface Pending {
  request_id: string;
  prompt: string;
  kind: string;
  details?: string;
  tool?: string;
}
export interface Session extends SessionSummary {
  sequence: number;
  messages: Message[];
  pending: Pending[];
  steps: RecordData;
  context: RecordData;
}
export interface Info {
  cwd: string;
  workspace: string;
  version: string;
  model: string;
  provider: string;
  active_session: string | null;
}
export interface Provider {
  id: string;
  display_name: string;
  prefix: string;
  default_base_url: string;
  api_key_required: boolean;
  base_url_editable: boolean;
  static_models: string[];
}
export interface Models {
  current: string;
  provider: string;
  providers: Provider[];
  models: string[];
  error: string | null;
  effort: {
    choices: string[];
    description: string;
    value: string;
    mechanism: string;
  };
}
export interface TeamAgent {
  id?: string;
  name?: string;
  role?: string;
  status?: string;
  tools?: string[];
  tool_names?: string[];
  system_prompt?: string;
  ticket_id?: string;
  cursor?: number;
  received?: number[];
  waiting?: {
    events: string[];
    reason: string;
    since: string;
    sender?: string;
    reply_to?: number;
    task_ids?: string[];
    ticket_id?: string;
    timeout?: number;
  };
  [key: string]: unknown;
}
export interface Ticket {
  id?: string;
  title?: string;
  status?: string;
  description?: string;
  assigned_to?: string;
  assignee?: string;
  result?: unknown;
  [key: string]: unknown;
}
export interface TeamEvent {
  id: number;
  kind: string;
  author: string;
  author_label?: string;
  recipient?: string;
  recipient_label?: string;
  thread_id?: number;
  reply_to?: number;
  message_type?: "request" | "reply" | "info";
  delivery?: string;
  content: string;
  created_at: string;
  ticket_id?: string;
  superseded_by?: number;
}
export interface Team {
  board: {
    agents?: Record<string, TeamAgent>;
    tickets?: Record<string, Ticket>;
    [key: string]: unknown;
  };
  live: boolean;
  events: TeamEvent[];
  next: number;
  has_more: boolean;
}
export interface Process {
  id: string;
  description: string;
  command: string;
  cwd: string;
  status: string;
  exit_code: number | null;
  runtime_seconds: number;
}
export interface ProcessOutput {
  id: string;
  output: string;
  discarded: number;
  truncated: boolean;
  status: string;
  exit_code: number | null;
}
export interface FileData {
  path: string;
  text: string;
  revision: string;
  binary: boolean;
}
export interface FileEntry {
  name: string;
  path: string;
  is_dir: boolean;
}
