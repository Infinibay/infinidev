import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { useEngine } from "@/state/store";

interface Entry {
  cmd: string;
  out: string;
  err: string;
  code: number | null; // null while running
}

/** Interactive terminal: shell commands the agent ran (read-only, from the
 *  event stream) plus a prompt where the user can run their own commands in the
 *  project directory. */
export function TerminalTab() {
  const { state, commands } = useEngine();
  const agentCmds = state.messages.filter((m) => m.kind === "tool" && m.toolName === "execute_command");
  const [history, setHistory] = useState<Entry[]>([]);
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [recall, setRecall] = useState<number | null>(null);
  const scroller = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight });
  }, [history, agentCmds.length, busy]);

  const run = async () => {
    const cmd = value.trim();
    if (!cmd || busy) return;
    setValue("");
    setRecall(null);
    const idx = history.length;
    setHistory((h) => [...h, { cmd, out: "", err: "", code: null }]);
    setBusy(true);
    try {
      const r = await commands.runCommand(cmd);
      setHistory((h) => h.map((e, i) => (i === idx ? { ...e, out: r.stdout, err: r.stderr, code: r.code } : e)));
    } catch (e) {
      setHistory((h) =>
        h.map((entry, i) => (i === idx ? { ...entry, err: String((e as Error).message), code: -1 } : entry)),
      );
    } finally {
      setBusy(false);
    }
  };

  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      void run();
    } else if (e.key === "ArrowUp") {
      // Recall previous commands.
      e.preventDefault();
      if (history.length === 0) return;
      const next = recall == null ? history.length - 1 : Math.max(0, recall - 1);
      setRecall(next);
      setValue(history[next].cmd);
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      if (recall == null) return;
      const next = recall + 1;
      if (next >= history.length) {
        setRecall(null);
        setValue("");
      } else {
        setRecall(next);
        setValue(history[next].cmd);
      }
    }
  };

  return (
    <div className="flex h-full flex-col bg-surface">
      <div ref={scroller} className="mono min-h-0 flex-1 space-y-3 overflow-y-auto p-3 text-[11px] leading-relaxed">
        {agentCmds.length > 0 && (
          <div className="space-y-3">
            <div className="text-[10px] uppercase tracking-wider text-fg-subtle">Agent commands</div>
            {agentCmds.map((c) => {
              const cmd = ((c.toolArgs as Record<string, unknown> | null)?.command as string) ?? "";
              return (
                <div key={c.id}>
                  <div className="text-accent">$ {cmd}</div>
                  <pre className="whitespace-pre-wrap text-fg-muted">{c.toolOutput ?? (c.toolDone ? "" : "…")}</pre>
                </div>
              );
            })}
            <div className="border-t border-fg/8" />
          </div>
        )}

        {agentCmds.length === 0 && history.length === 0 && (
          <div className="text-fg-subtle">
            Run shell commands in the project directory — try <span className="text-accent">ls</span> or{" "}
            <span className="text-accent">pwd</span>.
          </div>
        )}

        {history.map((e, i) => (
          <div key={i}>
            <div className="text-accent">$ {e.cmd}</div>
            {e.out && <pre className="whitespace-pre-wrap text-fg-muted">{e.out}</pre>}
            {e.err && <pre className="whitespace-pre-wrap text-danger">{e.err}</pre>}
            {e.code == null ? (
              <span className="text-fg-subtle">…</span>
            ) : e.code !== 0 ? (
              <span className="text-[10px] text-danger">exit {e.code}</span>
            ) : null}
          </div>
        ))}
      </div>

      <div className="flex items-center gap-2 border-t border-fg/10 px-3 py-2">
        <span className="mono text-accent">$</span>
        <input
          className="mono flex-1 bg-transparent text-[12px] text-fg outline-none placeholder:text-fg-subtle"
          placeholder={busy ? "running…" : "type a command and press Enter"}
          value={value}
          spellCheck={false}
          disabled={busy}
          autoComplete="off"
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={onKey}
        />
      </div>
    </div>
  );
}
