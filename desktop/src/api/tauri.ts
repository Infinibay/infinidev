/** Tauri-backed transport + commands — the real desktop path (embedded Rust engine). */
import { invoke, onAskPrompt, onEngineEvent } from "@/platform/tauri";
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
import type { Commands, ConfigPatch, Transport } from "./transport";

export const tauriTransport: Transport = {
  kind: "tauri",
  subscribe(cb: (ev: EngineEvent) => void) {
    return onEngineEvent((p) => cb(p as EngineEvent));
  },
  send(text: string, images?: string[]) {
    return invoke<void>("engine_send", { text, images: images ?? [] });
  },
  reset() {
    return invoke<void>("reset_session");
  },
  cancel() {
    return invoke<void>("cancel_turn");
  },
  subscribeAsk(cb: (p: AskPrompt) => void) {
    return onAskPrompt((p) => cb(p as AskPrompt));
  },
  answer(id: number, answer: string) {
    return invoke<void>("answer_prompt", { id, answer });
  },
};

export const tauriCommands: Commands = {
  projectPath: () => invoke<string>("project_path"),
  setProject: (path) => invoke<string>("set_project", { path }),
  getConfig: () => invoke<EngineConfig>("get_config"),
  setConfig: (patch: ConfigPatch) => invoke<EngineConfig>("set_config", { patch }),
  providers: () => invoke<ProviderDto[]>("providers"),
  listModels: (provider, apiKey, baseUrl) =>
    invoke<string[]>("list_provider_models", { provider, apiKey, baseUrl }),
  fsTree: (path) => invoke<RsTreeEntry[]>("fs_tree", { path }),
  fsRead: (path) => invoke<{ path: string; text: string; binary: boolean }>("fs_read", { path }),
  fsWrite: (path, text) => invoke<void>("fs_write", { path, text }),
  fsCreate: (path, dir) => invoke<void>("fs_create", { path, dir }),
  fsRename: (from, to) => invoke<void>("fs_rename", { from, to }),
  fsDelete: (path) => invoke<void>("fs_delete", { path }),
  search: (query) => invoke<RsSearchHit[]>("project_search", { query }),
  findings: (query) => invoke<Finding[]>("knowledge_search", { query, limit: 200 }),
  recordFinding: (topic, content, findingType) =>
    invoke<number>("knowledge_record", { topic, content, findingType }),
  gitDiff: (path) => invoke<string>("git_diff", { path }),
  runCommand: (command) =>
    invoke<{ stdout: string; stderr: string; code: number }>("run_command", { command }),
  getHistory: () => invoke<{ role: string; content: string }[]>("get_history"),
  notesList: () => invoke<NoteRow[]>("notes_list"),
  notesClear: () => invoke<void>("notes_clear"),
  bgList: () => invoke<BgTask[]>("bg_list"),
  bgStop: (id) => invoke<boolean>("bg_stop", { id }),
  docsLibraries: () => invoke<DocLibrary[]>("docs_libraries"),
  docsSections: (library) => invoke<DocSection[]>("docs_sections", { library }),
  docsRead: (library, section) => invoke<string>("docs_read", { library, section }),
  docsSearch: (query) => invoke<DocHit[]>("docs_search", { query }),
  docsFetch: (library, url, version) => invoke<number>("docs_fetch", { library, url, version }),
  docsDelete: (library) => invoke<void>("docs_delete", { library }),
};
