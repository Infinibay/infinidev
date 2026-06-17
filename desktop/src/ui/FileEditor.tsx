import { useEffect, useRef, useState } from "react";
import { useToast } from "@harbor/components/feedback/Toast";
import { CodeBlock } from "@harbor/components/dev/CodeBlock";
import type { Commands } from "@/api/transport";
import { DiffView } from "./workbench/DiffView";

/** Map a file extension to a syntax-highlighting language hint for CodeBlock. */
export function langFor(path: string): string {
  const ext = path.split(".").pop()?.toLowerCase() ?? "";
  const map: Record<string, string> = {
    ts: "typescript", tsx: "tsx", js: "javascript", jsx: "jsx", mjs: "javascript",
    rs: "rust", py: "python", go: "go", rb: "ruby", java: "java", kt: "kotlin",
    c: "c", h: "c", cpp: "cpp", cc: "cpp", hpp: "cpp", cs: "csharp",
    json: "json", yaml: "yaml", yml: "yaml", toml: "toml", md: "markdown",
    html: "html", css: "css", scss: "scss", sh: "bash", bash: "bash", sql: "sql",
  };
  return map[ext] ?? "text";
}

export function basename(p: string): string {
  return p.split("/").pop() || p;
}

type Mode = "view" | "edit" | "diff";

/** A single file editor: syntax-highlighted view, plain-text edit, and a git
 *  diff, with dirty tracking reported up via `onDirty`. Used as a tab in the
 *  main pane. Kept mounted while its tab is inactive so edits survive switching. */
export function FileEditor({
  path,
  commands,
  visible,
  onDirty,
}: {
  path: string;
  commands: Commands;
  visible: boolean;
  onDirty: (path: string, dirty: boolean) => void;
}) {
  const [text, setText] = useState<string | null>(null);
  const [orig, setOrig] = useState("");
  const [binary, setBinary] = useState(false);
  const [mode, setMode] = useState<Mode>("view");
  const [diff, setDiff] = useState<string | null>(null);
  const { push } = useToast();

  // In-file search (⌘F). Find works against the textarea, so opening it also
  // switches to edit mode where matches can be selected and scrolled into view.
  const taRef = useRef<HTMLTextAreaElement>(null);
  const findRef = useRef<HTMLInputElement>(null);
  const [find, setFind] = useState<{ open: boolean; q: string; idx: number }>({ open: false, q: "", idx: 0 });

  const matches = (() => {
    if (!find.q || text == null) return [] as number[];
    const hay = text.toLowerCase();
    const needle = find.q.toLowerCase();
    const out: number[] = [];
    let i = hay.indexOf(needle);
    while (i !== -1) {
      out.push(i);
      i = hay.indexOf(needle, i + Math.max(1, needle.length));
    }
    return out;
  })();

  const gotoMatch = (n: number) => {
    const ta = taRef.current;
    if (!ta || matches.length === 0) return;
    const idx = ((n % matches.length) + matches.length) % matches.length;
    setFind((f) => ({ ...f, idx }));
    const start = matches[idx];
    ta.focus();
    ta.setSelectionRange(start, start + find.q.length);
    // Approximate scroll-to: position the caret line near the top.
    const before = (text ?? "").slice(0, start).split("\n").length - 1;
    const lineHeight = 18;
    ta.scrollTop = Math.max(0, before * lineHeight - ta.clientHeight / 2);
  };

  const openFind = () => {
    setMode("edit");
    setFind((f) => ({ ...f, open: true }));
    setTimeout(() => findRef.current?.focus(), 0);
  };

  // Select the first match as the query changes (kept here so `matches` is fresh).
  useEffect(() => {
    if (find.open && find.q && matches.length) gotoMatch(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [find.q, find.open]);

  // ⌘F / Esc, only for the editor whose tab is active.
  useEffect(() => {
    if (!visible) return;
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "f") {
        e.preventDefault();
        openFind();
      } else if (e.key === "Escape" && find.open) {
        setFind((f) => ({ ...f, open: false }));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [visible, find.open]);

  useEffect(() => {
    let alive = true;
    commands
      .fsRead(path)
      .then((r) => {
        if (!alive) return;
        setBinary(r.binary);
        setText(r.text);
        setOrig(r.text);
      })
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [path, commands]);

  const dirty = text != null && text !== orig;
  const reported = useRef(false);
  useEffect(() => {
    if (reported.current !== dirty) {
      reported.current = dirty;
      onDirty(path, dirty);
    }
  }, [dirty, path, onDirty]);

  const save = async () => {
    if (text == null) return;
    try {
      await commands.fsWrite(path, text);
      setOrig(text);
      push({ title: "Saved", description: path, tone: "success" });
    } catch (e) {
      push({ title: "Save failed", description: String((e as Error).message), tone: "danger" });
    }
  };

  const showDiff = async () => {
    setMode("diff");
    setDiff(null);
    try {
      setDiff(await commands.gitDiff(path));
    } catch (e) {
      setDiff(`(could not compute diff: ${String((e as Error).message)})`);
    }
  };

  const ModeTab = ({ m, label }: { m: Mode; label: string }) => (
    <button
      onClick={() => (m === "diff" ? void showDiff() : setMode(m))}
      className={`rounded px-2 py-0.5 text-[11px] ${
        mode === m ? "bg-surface-3/60 text-fg" : "text-fg-muted hover:text-fg"
      }`}
    >
      {label}
    </button>
  );

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b border-fg/8 px-3 py-1.5">
        <span className="mono truncate text-xs text-fg-muted">{path}</span>
        {dirty && <span className="text-warning">●</span>}
        {!binary && (
          <div className="ml-auto flex items-center gap-1">
            <ModeTab m="view" label="View" />
            <ModeTab m="edit" label="Edit" />
            <ModeTab m="diff" label="Diff" />
          </div>
        )}
        <button
          onClick={() => void save()}
          disabled={!dirty}
          className={`${binary ? "ml-auto " : ""}rounded bg-accent/90 px-2 py-0.5 text-[11px] text-white disabled:opacity-40`}
        >
          Save
        </button>
      </div>
      {find.open && !binary && (
        <div className="flex items-center gap-2 border-b border-fg/8 bg-surface-2/40 px-3 py-1.5">
          <input
            ref={findRef}
            className="mono min-w-0 flex-1 bg-transparent text-[12px] text-fg outline-none placeholder:text-fg-subtle"
            placeholder="Find in file…"
            value={find.q}
            spellCheck={false}
            onChange={(e) => setFind((f) => ({ ...f, q: e.target.value, idx: 0 }))}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                gotoMatch(e.shiftKey ? find.idx - 1 : find.idx + 1);
              } else if (e.key === "Escape") {
                setFind((f) => ({ ...f, open: false }));
              }
            }}
          />
          <span className="shrink-0 text-[11px] text-fg-subtle">
            {matches.length ? `${Math.min(find.idx + 1, matches.length)}/${matches.length}` : "0/0"}
          </span>
          <button onClick={() => gotoMatch(find.idx - 1)} className="text-[11px] text-fg-muted hover:text-fg" aria-label="Previous match">
            ↑
          </button>
          <button onClick={() => gotoMatch(find.idx + 1)} className="text-[11px] text-fg-muted hover:text-fg" aria-label="Next match">
            ↓
          </button>
          <button onClick={() => setFind((f) => ({ ...f, open: false }))} className="text-[11px] text-fg-muted hover:text-fg" aria-label="Close find">
            ×
          </button>
        </div>
      )}
      {binary ? (
        <div className="p-4 text-xs text-fg-subtle">Binary file — not shown.</div>
      ) : mode === "edit" ? (
        <textarea
          ref={taRef}
          className="mono flex-1 resize-none bg-surface p-3 text-xs leading-relaxed text-fg outline-none"
          value={text ?? ""}
          spellCheck={false}
          autoFocus={visible && !find.open}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === "s") {
              e.preventDefault();
              void save();
            }
          }}
        />
      ) : mode === "diff" ? (
        <div className="min-h-0 flex-1 overflow-auto">
          {diff == null ? (
            <div className="p-4 text-xs text-fg-subtle">Computing diff…</div>
          ) : (
            <DiffView diff={diff} />
          )}
        </div>
      ) : (
        <div className="min-h-0 flex-1 overflow-auto" onDoubleClick={() => setMode("edit")}>
          <CodeBlock
            code={text ?? ""}
            lang={langFor(path)}
            showLineNumbers
            className="!m-0 min-h-full !rounded-none border-0 text-xs"
          />
        </div>
      )}
    </div>
  );
}
