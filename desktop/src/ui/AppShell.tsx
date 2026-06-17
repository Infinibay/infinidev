import { useCallback, useEffect, useRef, useState } from "react";
import { useAppState, useEngine } from "@/state/store";
import { isTauri, onMenuAction, pickFolder } from "@/platform/tauri";
import { Topbar } from "./Topbar";
import { MainPane } from "./MainPane";
import { Workbench, type WorkbenchTab } from "./workbench/Workbench";
import { SettingsModal } from "./modals/SettingsModal";
import { SearchPalette } from "./modals/SearchPalette";
import { KnowledgeModal } from "./modals/KnowledgeModal";
import { NotesModal } from "./modals/NotesModal";
import { BackgroundTasksModal } from "./modals/BackgroundTasksModal";
import { DocsModal } from "./modals/DocsModal";
import { DebugModal } from "./modals/DebugModal";
import { AboutModal } from "./modals/AboutModal";
import { AskModal } from "./modals/AskModal";
import { basename } from "./FileEditor";

type Modal =
  | null
  | "settings"
  | "search"
  | "about"
  | "knowledge"
  | "notes"
  | "background"
  | "docs"
  | "debug";

/** Render the conversation as Markdown and download it. */
function exportConversation(messages: { kind: string; text: string; toolName?: string; toolArgs?: unknown }[]) {
  const body = messages
    .map((m) => {
      switch (m.kind) {
        case "user":
          return `### You\n\n${m.text}`;
        case "agent":
          return `### Infinidev\n\n${m.text}`;
        case "system":
          return `_${m.text}_`;
        case "error":
          return `> ⚠️ ${m.text}`;
        case "tool": {
          const arg = (() => {
            const a = m.toolArgs as Record<string, unknown> | null;
            const v = a && (a.path ?? a.command ?? a.pattern ?? a.query);
            return typeof v === "string" ? ` \`${v}\`` : "";
          })();
          return `- 🔧 \`${m.toolName}\`${arg}`;
        }
        default:
          return m.text;
      }
    })
    .join("\n\n");
  const md = `# Infinidev conversation\n\n${body}\n`;
  const blob = new Blob([md], { type: "text/markdown" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "infinidev-conversation.md";
  a.click();
  URL.revokeObjectURL(url);
}

export function AppShell() {
  const s = useAppState();
  const { reset, commands, refreshProject, stop } = useEngine();
  const [modal, setModal] = useState<Modal>(null);
  const [tab, setTab] = useState<WorkbenchTab>("activity");

  // Main-pane tabs: "chat" plus one per open file. AppShell owns this because
  // files are opened from several places (tree, Changes, search).
  const [openFiles, setOpenFiles] = useState<string[]>([]);
  const [activeMain, setActiveMain] = useState<string>("chat");
  const [dirty, setDirty] = useState<Record<string, boolean>>({});

  const openFile = useCallback((path: string) => {
    setOpenFiles((f) => (f.includes(path) ? f : [...f, path]));
    setActiveMain(path);
  }, []);

  const onDirty = useCallback((path: string, isDirty: boolean) => {
    setDirty((d) => (d[path] === isDirty ? d : { ...d, [path]: isDirty }));
  }, []);

  const closeFile = useCallback(
    (path: string) => {
      if (dirty[path] && !window.confirm(`${basename(path)} has unsaved changes. Close anyway?`)) return;
      setOpenFiles((f) => {
        const next = f.filter((p) => p !== path);
        setActiveMain((a) => (a === path ? next[next.length - 1] ?? "chat" : a));
        return next;
      });
      setDirty((d) => {
        const { [path]: _drop, ...rest } = d;
        return rest;
      });
    },
    [dirty],
  );

  // The file the tree should highlight: the active main tab when it's a file.
  const selectedFile = activeMain === "chat" ? null : activeMain;

  // Native-menu actions (File/Edit/View). The handler is held in a ref so we
  // subscribe to the menu channel exactly once while always running the latest
  // closure (which closes over current commands/reset).
  const onMenu = useRef<(action: string) => void>(() => {});
  onMenu.current = (action: string) => {
    switch (action) {
      case "settings":
        setModal("settings");
        break;
      case "search":
        setModal("search");
        break;
      case "new_session":
      case "clear":
        reset();
        break;
      case "stop":
        stop();
        break;
      case "about":
        setModal("about");
        break;
      case "models":
        setModal("settings");
        break;
      case "show_knowledge":
        setModal("knowledge");
        break;
      case "show_notes":
        setModal("notes");
        break;
      case "show_background":
        setModal("background");
        break;
      case "show_docs":
        setModal("docs");
        break;
      case "show_debug":
        setModal("debug");
        break;
      case "export":
        exportConversation(s.messages);
        break;
      case "show_activity":
        setTab("activity");
        break;
      case "show_changes":
        setTab("changes");
        break;
      case "show_files":
        setTab("files");
        break;
      case "show_terminal":
        setTab("terminal");
        break;
      case "open_folder":
        void (async () => {
          const dir = await pickFolder();
          if (!dir) return;
          try {
            await commands.setProject(dir);
            await refreshProject();
            // Close any editors from the previous project and show the explorer.
            setOpenFiles([]);
            setDirty({});
            setActiveMain("chat");
            setTab("files");
          } catch {
            /* invalid folder — leave the current project */
          }
        })();
        break;
    }
  };

  useEffect(() => {
    if (!isTauri()) return;
    let alive = true;
    let unlisten = () => {};
    onMenuAction((action) => onMenu.current(action)).then((un) => {
      if (alive) unlisten = un;
      else un();
    });
    return () => {
      alive = false;
      unlisten();
    };
  }, []);

  // Slash commands in the composer route through the same handler as the menu.
  const runCommand = useCallback((action: string) => onMenu.current(action), []);

  // Auto-focus the Activity tab when a turn starts (unless the user is in Files).
  useEffect(() => {
    if (s.busy) setTab((t) => (t === "files" ? t : "activity"));
  }, [s.busy]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setModal("search");
      } else if (mod && e.key === ",") {
        e.preventDefault();
        setModal("settings");
      } else if (mod && e.key === ".") {
        e.preventDefault();
        onMenu.current("stop");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="fixed inset-0 flex flex-col text-fg">
      <Topbar
        onOpenSettings={() => setModal("settings")}
        onOpenSearch={() => setModal("search")}
        onAction={runCommand}
      />
      <div className="flex min-h-0 flex-1 overflow-hidden">
        {/* The main pane (Chat + open file editors as tabs) sits on the
            translucent canvas; the workbench is an opaque elevated panel beside it. */}
        <main className="min-w-0 flex-1">
          <MainPane
            openFiles={openFiles}
            active={activeMain}
            setActive={setActiveMain}
            closeFile={closeFile}
            dirty={dirty}
            onDirty={onDirty}
            onCommand={runCommand}
          />
        </main>
        <aside className="w-[42%] min-w-[340px] max-w-[600px] shrink-0 border-l border-fg/10 bg-surface-1/80 shadow-[-12px_0_32px_-16px_rgba(0,0,0,0.55)] backdrop-blur-sm">
          <Workbench tab={tab} setTab={setTab} selectedFile={selectedFile} openFile={openFile} />
        </aside>
      </div>
      <SettingsModal open={modal === "settings"} onClose={() => setModal(null)} />
      <SearchPalette open={modal === "search"} onClose={() => setModal(null)} openFile={openFile} />
      <KnowledgeModal open={modal === "knowledge"} onClose={() => setModal(null)} />
      <NotesModal open={modal === "notes"} onClose={() => setModal(null)} />
      <BackgroundTasksModal open={modal === "background"} onClose={() => setModal(null)} />
      <DocsModal open={modal === "docs"} onClose={() => setModal(null)} />
      <DebugModal open={modal === "debug"} onClose={() => setModal(null)} />
      <AboutModal open={modal === "about"} onClose={() => setModal(null)} />
      <AskModal />
    </div>
  );
}
