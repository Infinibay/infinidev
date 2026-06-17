import { useRef, useState, type KeyboardEvent } from "react";
import { Button } from "@harbor/components/buttons/Button";
import { Z } from "@harbor/lib/z";
import { useEngine } from "@/state/store";

/** Read picked image files into `data:` URLs the engine can attach to the turn. */
function readAsDataUrls(files: FileList): Promise<string[]> {
  return Promise.all(
    [...files]
      .filter((f) => f.type.startsWith("image/"))
      .map(
        (f) =>
          new Promise<string>((resolve, reject) => {
            const r = new FileReader();
            r.onload = () => resolve(String(r.result));
            r.onerror = () => reject(r.error);
            r.readAsDataURL(f);
          }),
      ),
  );
}

interface PaletteItem {
  cmd: string;
  desc: string;
  /** UI/navigation commands dispatch an action. */
  action?: string;
  /** Flow commands wrap the rest of the input into a directive prompt. */
  wrap?: (task: string) => string;
  /** Whether the flow needs a task argument (false for self-contained flows). */
  needsArg?: boolean;
}

/** UI/navigation slash commands. `action` is dispatched on select. */
const COMMANDS: PaletteItem[] = [
  { cmd: "/models", desc: "Pick a model", action: "models" },
  { cmd: "/settings", desc: "Open settings", action: "settings" },
  { cmd: "/files", desc: "Open the file explorer", action: "show_files" },
  { cmd: "/changes", desc: "Show changed files", action: "show_changes" },
  { cmd: "/activity", desc: "Show agent activity", action: "show_activity" },
  { cmd: "/terminal", desc: "Open the terminal", action: "show_terminal" },
  { cmd: "/search", desc: "Search the project", action: "search" },
  { cmd: "/knowledge", desc: "Browse project findings", action: "show_knowledge" },
  { cmd: "/export", desc: "Export the conversation", action: "export" },
  { cmd: "/open", desc: "Open a folder", action: "open_folder" },
  { cmd: "/stop", desc: "Stop the current turn", action: "stop" },
  { cmd: "/clear", desc: "Clear the conversation", action: "clear" },
  { cmd: "/help", desc: "Show all commands", action: "help" },
];

/** Flow commands shape the next turn (work with the existing engine — they just
 *  wrap the task in a directive). Mirrors the TUI's /plan, /think, … modes. */
const FLOWS: PaletteItem[] = [
  { cmd: "/plan", desc: "Plan the task, then carry it out", needsArg: true,
    wrap: (t) => `Produce a concise, numbered step-by-step plan for the task below, then carry it out:\n\n${t}` },
  { cmd: "/think", desc: "Gather context deeply before acting", needsArg: true,
    wrap: (t) => `Think carefully and gather the full context you need (read the relevant files) before acting on:\n\n${t}` },
  { cmd: "/explore", desc: "Explore and explain in depth", needsArg: true,
    wrap: (t) => `Explore the codebase and explain in depth, citing the relevant files:\n\n${t}` },
  { cmd: "/refactor", desc: "Refactor without changing behavior", needsArg: true,
    wrap: (t) => `Refactor the following for clarity, structure, and naming while preserving behavior. Show what you changed:\n\n${t}` },
  { cmd: "/init", desc: "Explore & document this project", needsArg: false,
    wrap: () => `Explore this project end-to-end and create or update documentation (a README or AGENTS file) describing what it does, its structure, and how to build, run, and test it.` },
];

const PALETTE: PaletteItem[] = [...FLOWS, ...COMMANDS];

const flowFor = (text: string): { flow: PaletteItem; task: string } | null => {
  const tok = text.trim().split(/\s/)[0].toLowerCase();
  const flow = FLOWS.find((f) => f.cmd === tok);
  if (!flow) return null;
  return { flow, task: text.trim().slice(tok.length).trim() };
};

interface Props {
  /** Route UI commands (settings, tab switches, open folder) to the shell. */
  onCommand?: (action: string) => void;
}

export function Composer({ onCommand }: Props) {
  const { state, send, commands, setModel, reset, stop } = useEngine();
  const [value, setValue] = useState("");
  const [hi, setHi] = useState(0); // highlighted row in the active popup
  const [dismissed, setDismissed] = useState(false); // Esc-hid the slash list
  const [picker, setPicker] = useState<null | "models">(null);
  const [models, setModels] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [attachments, setAttachments] = useState<string[]>([]);
  const fileInput = useRef<HTMLInputElement>(null);

  const addFiles = async (files: FileList | null) => {
    if (!files?.length) return;
    try {
      const urls = await readAsDataUrls(files);
      if (urls.length) setAttachments((a) => [...a, ...urls]);
    } catch {
      /* ignore unreadable files */
    }
  };

  const trimmed = value.trimStart();
  // The palette is for picking a command; once the user types a space (a flow's
  // task argument), it closes so Enter submits instead of re-selecting.
  const hasArgs = /\s/.test(trimmed);
  const slashOpen = picker === null && !dismissed && trimmed.startsWith("/") && !hasArgs;
  const matches = slashOpen
    ? PALETTE.filter((c) => c.cmd.startsWith(trimmed.toLowerCase()))
    : [];
  const filteredModels =
    picker === "models"
      ? models.filter((m) => m.toLowerCase().includes(value.trim().toLowerCase()))
      : [];

  const openModels = async () => {
    setValue("");
    setHi(0);
    setPicker("models");
    setLoading(true);
    try {
      const cfg = await commands.getConfig();
      const list = await commands.listModels(cfg.provider, cfg.api_key, cfg.base_url);
      setModels(list);
      const idx = list.indexOf(cfg.model);
      setHi(idx >= 0 ? idx : 0);
    } catch {
      setModels([]);
    } finally {
      setLoading(false);
    }
  };

  const chooseModel = async (m: string) => {
    setPicker(null);
    setValue("");
    setModel(m);
    try {
      await commands.setConfig({ model: m });
    } catch {
      /* keep the optimistic selection */
    }
  };

  const runCmd = (action: string) => {
    switch (action) {
      case "models":
        void openModels();
        break;
      case "clear":
        reset();
        setValue("");
        break;
      case "stop":
        stop();
        setValue("");
        break;
      case "help":
        setValue("/");
        setDismissed(false);
        setHi(0);
        break;
      default:
        onCommand?.(action);
        setValue("");
    }
  };

  // Select a palette item: run a nav action, or start/submit a flow.
  const onSelect = (item: PaletteItem) => {
    if (item.action) {
      runCmd(item.action);
      return;
    }
    if (item.needsArg) {
      setValue(`${item.cmd} `); // user types the task, then Enter
      setHi(0);
    } else if (item.wrap) {
      if (!state.busy) send(item.wrap(""));
      setValue("");
    }
  };

  const submit = () => {
    const t = value.trim();
    const imgs = attachments;
    if ((!t && !imgs.length) || state.busy) return;
    // A flow command (e.g. "/plan add tests") wraps the task into a directive.
    const f = flowFor(t);
    if (f) {
      if (f.flow.needsArg && !f.task) return; // wait for a task
      send(f.flow.wrap!(f.task), imgs.length ? imgs : undefined);
      setValue("");
      setAttachments([]);
      return;
    }
    send(t, imgs.length ? imgs : undefined);
    setValue("");
    setAttachments([]);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (picker === "models") {
      const n = filteredModels.length;
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setHi((i) => (n ? (i + 1) % n : 0));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setHi((i) => (n ? (i - 1 + n) % n : 0));
      } else if (e.key === "Enter") {
        e.preventDefault();
        const m = filteredModels[Math.min(hi, n - 1)];
        if (m) void chooseModel(m);
      } else if (e.key === "Escape") {
        e.preventDefault();
        setPicker(null);
        setValue("");
      }
      return;
    }

    if (matches.length) {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setHi((i) => (i + 1) % matches.length);
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setHi((i) => (i - 1 + matches.length) % matches.length);
      } else if (e.key === "Enter" || e.key === "Tab") {
        e.preventDefault();
        onSelect(matches[Math.min(hi, matches.length - 1)]);
      } else if (e.key === "Escape") {
        e.preventDefault();
        setDismissed(true);
      }
      return;
    }

    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const activeLen = picker === "models" ? filteredModels.length : matches.length;
  const sel = Math.min(hi, Math.max(0, activeLen - 1));

  return (
    <div className="relative px-4 pb-4 pt-2">
      {/* ── Model picker popup ── */}
      {picker === "models" && (
        <div
          style={{ zIndex: Z.POPOVER }}
          className="absolute bottom-full left-4 right-4 mb-2 overflow-hidden rounded-xl border border-fg/12 bg-surface-2/95 shadow-harbor-lg backdrop-blur"
        >
          <div className="flex items-center justify-between border-b border-fg/8 px-3 py-2 text-[11px] text-fg-subtle">
            <span className="font-semibold uppercase tracking-wider">Select model</span>
            <span>{loading ? "loading…" : `${filteredModels.length} found`} · Esc to cancel</span>
          </div>
          <div className="max-h-60 overflow-y-auto py-1">
            {!loading && filteredModels.length === 0 && (
              <div className="px-3 py-2 text-xs text-fg-subtle">No models found.</div>
            )}
            {filteredModels.map((m, i) => (
              <button
                key={m}
                onClick={() => void chooseModel(m)}
                onMouseEnter={() => setHi(i)}
                className={`mono flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs ${
                  i === sel ? "bg-accent/20 text-fg" : "text-fg-muted hover:bg-surface-3/60"
                }`}
              >
                <span className="truncate">{m}</span>
                {m === state.model && <span className="ml-auto text-[10px] text-accent">current</span>}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* ── Slash-command palette ── */}
      {matches.length > 0 && (
        <div
          style={{ zIndex: Z.POPOVER }}
          className="absolute bottom-full left-4 right-4 mb-2 overflow-hidden rounded-xl border border-fg/12 bg-surface-2/95 shadow-harbor-lg backdrop-blur"
        >
          <div className="max-h-72 overflow-y-auto py-1">
            {matches.map((c, i) => (
              <button
                key={c.cmd}
                onClick={() => onSelect(c)}
                onMouseEnter={() => setHi(i)}
                className={`flex w-full items-center gap-3 px-3 py-1.5 text-left ${
                  i === sel ? "bg-accent/20" : "hover:bg-surface-3/60"
                }`}
              >
                <span className={`mono text-xs ${i === sel ? "text-fg" : "text-accent"}`}>{c.cmd}</span>
                <span className="truncate text-xs text-fg-muted">{c.desc}</span>
                {c.wrap && <span className="ml-auto text-[9px] uppercase tracking-wider text-fg-subtle">flow</span>}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* ── Attachment thumbnails ── */}
      {attachments.length > 0 && (
        <div className="mb-2 flex flex-wrap gap-2">
          {attachments.map((src, i) => (
            <div key={i} className="group relative h-14 w-14 overflow-hidden rounded-md border border-fg/15">
              <img src={src} alt={`attachment ${i + 1}`} className="h-full w-full object-cover" />
              <button
                onClick={() => setAttachments((a) => a.filter((_, j) => j !== i))}
                className="absolute right-0 top-0 hidden h-4 w-4 items-center justify-center bg-surface/80 text-[10px] text-fg group-hover:flex"
                title="Remove"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      <input
        ref={fileInput}
        type="file"
        accept="image/*"
        multiple
        className="hidden"
        onChange={(e) => {
          void addFiles(e.target.files);
          e.target.value = "";
        }}
      />

      <div className="flex items-end gap-2 rounded-xl border border-fg/15 bg-surface-2/70 p-3 shadow-harbor-sm focus-within:border-accent/50">
        {picker === null && (
          <button
            onClick={() => fileInput.current?.click()}
            disabled={state.busy}
            title="Attach images"
            className="grid h-8 w-8 shrink-0 place-items-center rounded-md text-fg-muted hover:bg-surface-3/60 hover:text-fg disabled:opacity-40"
          >
            📎
          </button>
        )}
        <textarea
          className="max-h-60 min-h-[68px] flex-1 resize-none bg-transparent px-2 py-1.5 text-[15px] leading-relaxed text-fg outline-none placeholder:text-fg-subtle"
          rows={2}
          value={value}
          placeholder={
            picker === "models"
              ? "Filter models…"
              : state.busy
                ? "Working…"
                : "Ask Infinidev to build, fix, or explain — or type / for commands"
          }
          disabled={state.busy && picker === null}
          autoFocus
          onChange={(e) => {
            setValue(e.target.value);
            setHi(0);
            setDismissed(false);
          }}
          onKeyDown={onKeyDown}
        />
        {picker === "models" ? (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              setPicker(null);
              setValue("");
            }}
          >
            Cancel
          </Button>
        ) : state.busy ? (
          <Button variant="destructive" size="sm" onClick={stop} title="Stop the current turn (⌘.)">
            Stop
          </Button>
        ) : (
          <Button variant="primary" size="sm" onClick={submit} disabled={!value.trim() && attachments.length === 0}>
            Send
          </Button>
        )}
      </div>
      <div className="mt-1 px-1 text-[10px] text-fg-subtle">
        {picker === "models"
          ? "↑↓ to navigate · Enter to select · Esc to cancel"
          : "Enter to send · Shift+Enter for newline · / for commands"}
      </div>
    </div>
  );
}
