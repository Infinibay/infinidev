/**
 * Demo transport + commands for browser/preview (no backend). Replays a
 * realistic scripted turn so the whole UI is exercisable, and serves canned
 * project data for the explorer / search / model picker.
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
import type { Commands, ConfigPatch, Transport } from "./transport";

// Mutable so the demo reflects setConfig edits across the session (like the real backend).
let demoConfig: EngineConfig = {
  provider: "ollama",
  model: "demo-model",
  api_key: null,
  base_url: "",
  temperature: 0.2,
  max_iterations: 25,
  planning: true,
  orchestrate: true,
  review: true,
  force_manual_tools: false,
  confirm_commands: false,
  confirm_writes: false,
};

let listener: (ev: EngineEvent) => void = () => {};
let askListener: (p: AskPrompt) => void = () => {};
let askResolver: ((answer: string) => void) | null = null;
let askId = 0;
let timers: ReturnType<typeof setTimeout>[] = [];

function clearTimers() {
  timers.forEach(clearTimeout);
  timers = [];
}

export const demoTransport: Transport = {
  kind: "demo",
  async subscribe(cb) {
    listener = cb;
    // Guard against React StrictMode's mount→cleanup→mount race: only detach
    // if we're still the active listener (a later subscribe may have replaced us).
    return () => {
      if (listener === cb) listener = () => {};
    };
  },
  async reset() {
    clearTimers();
  },
  async cancel() {
    clearTimers();
    if (askResolver) {
      askResolver("");
      askResolver = null;
    }
    listener({ type: "notify", speaker: "infinidev", text: "Turn cancelled.", kind: "system" });
    listener({ type: "turn_end", result: "" });
  },
  async subscribeAsk(cb) {
    askListener = cb;
    return () => {
      if (askListener === cb) askListener = () => {};
    };
  },
  async answer(_id: number, answer: string) {
    if (askResolver) {
      const r = askResolver;
      askResolver = null;
      r(answer);
    }
  },
  async send(text: string, images?: string[]) {
    clearTimers();
    const imgNote = images?.length ? ` (with ${images.length} image${images.length === 1 ? "" : "s"})` : "";

    // When command-confirmation is enabled, demo the interactive gate before
    // doing any work — wait for the user to allow/deny.
    if (demoConfig.confirm_commands) {
      listener({ type: "phase", phase: "working" });
      const ans = await new Promise<string>((resolve) => {
        askResolver = resolve;
        askListener({ id: ++askId, prompt: "Allow Infinidev to run this shell command?\n\nls -la", kind: "permission" });
      });
      if (ans.trim() !== "allow") {
        listener({ type: "notify", speaker: "infinidev", text: "Command denied by the user — stopping here.", kind: "system" });
        listener({ type: "turn_end", result: "" });
        return;
      }
    }

    let t = 0;
    const at = (delta: number, ev: EngineEvent) => {
      t += delta;
      timers.push(setTimeout(() => listener(ev), t));
    };
    const words = (s: string, ev: (w: string) => EngineEvent) =>
      s.split(" ").forEach((w) => at(38, ev(`${w} `)));

    at(40, { type: "phase", phase: "working" });
    at(120, { type: "decision", kind: "escalate", detail: "User wants a new example file created in the project root." });
    at(80, { type: "notify", speaker: "infinidev", text: "Voy a crear un archivo de ejemplo — arranco con el análisis.", kind: "preview" });
    at(60, {
      type: "plan",
      steps: [
        "Inspect the project layout",
        "Identify the language and entry points",
        "Summarize what the project does",
      ],
    });
    at(40, { type: "step_start", step: 1, max: 25 });
    words("Let me look at the project structure first.", (w) => ({ type: "stream_chunk", chunk: w }));
    at(250, { type: "tool_call", id: "c1", name: "list_directory", args: { path: "." } });
    at(450, { type: "tool_result", id: "c1", name: "list_directory", ok: true, output: "src/\nREADME.md\nCargo.toml" });
    at(150, { type: "step_start", step: 2, max: 25 });
    at(150, { type: "tool_call", id: "c2", name: "create_file", args: { path: "example.txt", content: "hello from infinidev" } });
    at(450, { type: "tool_result", id: "c2", name: "create_file", ok: true, output: "Wrote example.txt (20 bytes)." });
    at(40, { type: "file_change", path: "example.txt", action: "create" });
    at(150, { type: "step_start", step: 3, max: 25 });
    words(`Done — I created example.txt. (demo reply to: "${text}"${imgNote})`, (w) => ({ type: "stream_chunk", chunk: w }));
    at(120, { type: "usage", prompt_tokens: 1234, completion_tokens: 212, total_tokens: 1446, cost_usd: null });
    at(250, { type: "review", status: "approve", notes: [] });
    at(40, { type: "phase", phase: "idle" });
    at(40, { type: "turn_end", result: "I created example.txt." });
  },
};

const DEMO_TREE: Record<string, RsTreeEntry[]> = {
  "": [
    { name: "src", path: "src", is_dir: true },
    { name: "crates", path: "crates", is_dir: true },
    { name: "README.md", path: "README.md", is_dir: false },
    { name: "Cargo.toml", path: "Cargo.toml", is_dir: false },
  ],
  src: [{ name: "main.rs", path: "src/main.rs", is_dir: false }],
  crates: [{ name: "infinidev-llm", path: "crates/infinidev-llm", is_dir: true }],
};

const parentOf = (p: string) => p.split("/").slice(0, -1).join("/");
const nameOf = (p: string) => p.split("/").pop() || p;
const sortTree = (a: RsTreeEntry, b: RsTreeEntry) =>
  a.is_dir === b.is_dir ? a.name.localeCompare(b.name) : a.is_dir ? -1 : 1;

export const demoCommands: Commands = {
  projectPath: async () => "/demo/project",
  setProject: async (path: string) => path,
  getConfig: async () => ({ ...demoConfig }),
  setConfig: async (patch: ConfigPatch) => {
    demoConfig = {
      ...demoConfig,
      ...(patch.provider !== undefined && { provider: patch.provider }),
      ...(patch.model !== undefined && { model: patch.model }),
      ...(patch.api_key !== undefined && { api_key: patch.api_key }),
      ...(patch.base_url !== undefined && { base_url: patch.base_url }),
      ...(patch.temperature !== undefined && { temperature: patch.temperature }),
      ...(patch.max_iterations !== undefined && { max_iterations: patch.max_iterations }),
      ...(patch.planning !== undefined && { planning: patch.planning }),
      ...(patch.orchestrate !== undefined && { orchestrate: patch.orchestrate }),
      ...(patch.review !== undefined && { review: patch.review }),
      ...(patch.force_manual_tools !== undefined && { force_manual_tools: patch.force_manual_tools }),
      ...(patch.confirm_commands !== undefined && { confirm_commands: patch.confirm_commands }),
      ...(patch.confirm_writes !== undefined && { confirm_writes: patch.confirm_writes }),
    };
    return { ...demoConfig };
  },
  providers: async (): Promise<ProviderDto[]> => [
    { id: "ollama", display_name: "Ollama (Local)", api_key_required: false, base_url_editable: true, default_base_url: "http://localhost:11434" },
    { id: "openai", display_name: "OpenAI", api_key_required: true, base_url_editable: false, default_base_url: "https://api.openai.com/v1" },
    { id: "anthropic", display_name: "Claude (Anthropic)", api_key_required: true, base_url_editable: false, default_base_url: "https://api.anthropic.com" },
  ],
  listModels: async () => ["demo-model", "qwen2.5-coder:7b", "gemma4:e4b"],
  fsTree: async (path: string) => DEMO_TREE[path] ?? [],
  fsRead: async (path: string) => ({
    path,
    text: `// ${path}\n// (demo file content)\nfn main() {\n    println!("hello from infinidev");\n}\n`,
    binary: false,
  }),
  fsWrite: async () => {},
  fsCreate: async (path: string, dir: boolean) => {
    const parent = parentOf(path);
    (DEMO_TREE[parent] ??= []).push({ name: nameOf(path), path, is_dir: dir });
    DEMO_TREE[parent].sort(sortTree);
    if (dir) DEMO_TREE[path] = [];
  },
  fsRename: async (from: string, to: string) => {
    const parent = parentOf(from);
    const list = DEMO_TREE[parent] ?? [];
    const e = list.find((x) => x.path === from);
    if (e) {
      e.name = nameOf(to);
      e.path = to;
      list.sort(sortTree);
    }
    if (DEMO_TREE[from]) {
      DEMO_TREE[to] = DEMO_TREE[from];
      delete DEMO_TREE[from];
    }
  },
  fsDelete: async (path: string) => {
    const parent = parentOf(path);
    DEMO_TREE[parent] = (DEMO_TREE[parent] ?? []).filter((x) => x.path !== path);
    delete DEMO_TREE[path];
  },
  search: async (query: string): Promise<RsSearchHit[]> => [
    { path: "README.md", line: 1, text: `# Infinidev (demo match for "${query}")` },
    { path: "src/main.rs", line: 3, text: `println!("hello"); // ${query}` },
  ],
  findings: async (query: string): Promise<Finding[]> => {
    const q = query.trim().toLowerCase();
    return DEMO_FINDINGS.filter(
      (f) => !q || f.topic.toLowerCase().includes(q) || f.content.toLowerCase().includes(q),
    );
  },
  runCommand: async (command: string) => {
    // Canned responses so the interactive terminal is exercisable in the browser.
    const c = command.trim();
    if (c === "ls" || c === "ls -la" || c.startsWith("ls"))
      return { stdout: "Cargo.toml\nREADME.md\ncrates/\nsrc/\n", stderr: "", code: 0 };
    if (c.startsWith("echo ")) return { stdout: c.slice(5) + "\n", stderr: "", code: 0 };
    if (c === "pwd") return { stdout: "/demo/project\n", stderr: "", code: 0 };
    return { stdout: `(demo) ran: ${command}\n`, stderr: "", code: 0 };
  },
  recordFinding: async (topic: string, content: string, findingType: string): Promise<number> => {
    const id = (DEMO_FINDINGS[0]?.id ?? 0) + 1;
    DEMO_FINDINGS.unshift({
      id,
      topic,
      content,
      finding_type: findingType || "observation",
      confidence: 0.7,
      created_at: "2026-06-16 13:00:00",
    });
    return id;
  },
  getHistory: async () => [],
  notesList: async (): Promise<NoteRow[]> => DEMO_NOTES.slice(),
  notesClear: async () => {
    DEMO_NOTES.length = 0;
  },
  bgList: async (): Promise<BgTask[]> => DEMO_BG.slice(),
  bgStop: async (id: number): Promise<boolean> => {
    const t = DEMO_BG.find((x) => x.id === id);
    if (t && t.running) {
      t.running = false;
      t.status = "exited (137)";
      t.exit_code = 137;
      return true;
    }
    return false;
  },
  docsLibraries: async (): Promise<DocLibrary[]> =>
    [...new Set(DEMO_DOCS.map((d) => d.library))].map((library) => ({
      library,
      version: "latest",
      sections: DEMO_DOCS.filter((d) => d.library === library).length,
    })),
  docsSections: async (library: string): Promise<DocSection[]> =>
    DEMO_DOCS.filter((d) => d.library === library).map((d, i) => ({
      section_title: d.section_title,
      section_order: i,
      source_url: d.source_url,
    })),
  docsRead: async (library: string, section: string): Promise<string> =>
    DEMO_DOCS.find((d) => d.library === library && d.section_title === section)?.content ?? "",
  docsSearch: async (query: string): Promise<DocHit[]> => {
    const q = query.trim().toLowerCase();
    return DEMO_DOCS.filter(
      (d) => !q || d.section_title.toLowerCase().includes(q) || d.content.toLowerCase().includes(q),
    ).map((d) => ({ library: d.library, section_title: d.section_title, snippet: d.content.slice(0, 140) }));
  },
  docsFetch: async (library: string, url: string): Promise<number> => {
    DEMO_DOCS.push({
      library,
      section_title: "Overview",
      content: `(demo) fetched ${url} and cached it under "${library}".`,
      source_url: url,
    });
    return 1;
  },
  docsDelete: async (library: string): Promise<void> => {
    for (let i = DEMO_DOCS.length - 1; i >= 0; i--) if (DEMO_DOCS[i].library === library) DEMO_DOCS.splice(i, 1);
  },
  gitDiff: async (path: string): Promise<string> => {
    if (path === "example.txt") {
      return `--- /dev/null\n+++ b/example.txt\n@@ -0,0 +1,1 @@\n+hello from infinidev\n`;
    }
    return `--- a/${path}\n+++ b/${path}\n@@ -1,4 +1,5 @@\n // ${path}\n // (demo file content)\n fn main() {\n-    println!("hello from infinidev");\n+    println!("hello from infinidev");\n+    // edited by the agent\n }\n`;
  },
};

const DEMO_NOTES: NoteRow[] = [
  { id: 1, note: "The build uses `cargo test -p infinidev-engine` for the engine suite.", created_at: "2026-06-16 12:10:00" },
  { id: 2, note: "Vapor-pressure cascade prefers PubChem before Ambrose-Walton.", created_at: "2026-06-16 12:25:00" },
  { id: 3, note: "Prefer subpath imports (`@chem/engine/io`) for tree-shaking.", created_at: "2026-06-16 12:40:00" },
];

const DEMO_BG: BgTask[] = [
  {
    id: 1,
    command: "npm run dev",
    description: "vite dev server",
    status: "running",
    running: true,
    exit_code: null,
    output_tail: "VITE v5.4.2  ready in 412 ms\n➜  Local:   http://localhost:5173/\n➜  Network: use --host to expose",
  },
  {
    id: 2,
    command: "cargo watch -x check",
    description: "cargo watch",
    status: "exited (0)",
    running: false,
    exit_code: 0,
    output_tail: "[Running 'cargo check']\n   Compiling infinidev-engine\n    Finished dev [unoptimized] target(s) in 3.1s\n[Finished running. Exit status: 0]",
  },
];

const DEMO_DOCS: { library: string; section_title: string; content: string; source_url: string }[] = [
  { library: "tokio", section_title: "Spawning", content: "tokio::spawn runs a future on the runtime; it returns a JoinHandle you can await for the task's result.", source_url: "https://docs.rs/tokio" },
  { library: "tokio", section_title: "Runtime", content: "Build a multi-threaded runtime with tokio::runtime::Builder::new_multi_thread().enable_all().build().", source_url: "https://docs.rs/tokio" },
  { library: "tokio", section_title: "select!", content: "tokio::select! races several async branches and runs the first to complete, cancelling the rest.", source_url: "https://docs.rs/tokio" },
  { library: "serde", section_title: "Derive", content: "Add #[derive(Serialize, Deserialize)] to a struct to make it (de)serializable across formats.", source_url: "https://serde.rs" },
];

const DEMO_FINDINGS: Finding[] = [
  { id: 3, topic: "Agent loop", content: "The tool-calling loop lives in crates/infinidev-engine/src/engine.rs; it streams EngineEvents to the host.", finding_type: "pattern", confidence: 0.9, created_at: "2026-06-16 12:00:00" },
  { id: 2, topic: "Providers", content: "Provider registry + pricing is in infinidev-llm; PROVIDERS lists Ollama/OpenAI/Anthropic.", finding_type: "api", confidence: 0.8, created_at: "2026-06-16 11:40:00" },
  { id: 1, topic: "Findings store", content: "Knowledge is SQLite at <ws>/.infinidev/knowledge.db; substring search for now.", finding_type: "decision", confidence: 0.7, created_at: "2026-06-16 11:10:00" },
];
