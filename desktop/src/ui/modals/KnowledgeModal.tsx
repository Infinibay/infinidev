import { useEffect, useRef, useState } from "react";
import { Dialog } from "@harbor/components/overlays/Dialog";
import { TextField } from "@harbor/components/inputs/TextField";
import { Badge } from "@harbor/components/display/Badge";
import { Button } from "@harbor/components/buttons/Button";
import { useToast } from "@harbor/components/feedback/Toast";
import { useEngine } from "@/state/store";
import type { Finding } from "@/api/engineEvent";

interface Props {
  open: boolean;
  onClose: () => void;
}

/** Browser for the project knowledge base (`infinidev-knowledge`): the durable
 *  findings the agent records as it learns the codebase. Mirrors the TUI's
 *  findings browser — searchable list on the left, full content on the right. */
export function KnowledgeModal({ open, onClose }: Props) {
  const { commands, state } = useEngine();
  const { push } = useToast();
  const [q, setQ] = useState("");
  const [items, setItems] = useState<Finding[]>([]);
  const [sel, setSel] = useState<Finding | null>(null);
  const [busy, setBusy] = useState(false);
  const [adding, setAdding] = useState(false);
  const [topic, setTopic] = useState("");
  const [content, setContent] = useState("");
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const reload = () => {
    setBusy(true);
    commands
      .findings(q.trim())
      .then((f) => {
        setItems(f);
        setSel((cur) => cur ?? f[0] ?? null);
      })
      .catch(() => setItems([]))
      .finally(() => setBusy(false));
  };

  const saveFinding = async () => {
    if (!topic.trim() || !content.trim()) return;
    try {
      await commands.recordFinding(topic.trim(), content.trim(), "observation");
      push({ title: "Finding recorded", tone: "success" });
      setTopic("");
      setContent("");
      setAdding(false);
      reload();
    } catch (e) {
      push({ title: "Could not record", description: String((e as Error).message), tone: "danger" });
    }
  };

  // Load (and reload on each keystroke, debounced) while the dialog is open.
  useEffect(() => {
    if (!open) return;
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      setBusy(true);
      commands
        .findings(q.trim())
        .then((f) => {
          setItems(f);
          setSel((cur) => cur ?? f[0] ?? null);
        })
        .catch(() => setItems([]))
        .finally(() => setBusy(false));
    }, 200);
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, [open, q, commands, state.projectNonce]);

  useEffect(() => {
    if (!open) {
      setQ("");
      setSel(null);
      setItems([]);
      setAdding(false);
      setTopic("");
      setContent("");
    }
  }, [open]);

  return (
    <Dialog open={open} onClose={onClose} size="lg" title="Knowledge — project findings">
      <div className="flex items-center gap-2">
        <div className="flex-1">
          <TextField placeholder="Filter findings…" value={q} onChange={(e) => setQ(e.target.value)} />
        </div>
        <Button variant={adding ? "ghost" : "secondary"} size="sm" onClick={() => setAdding((a) => !a)}>
          {adding ? "Cancel" : "Add finding"}
        </Button>
      </div>

      {adding && (
        <div className="mt-2 space-y-2 rounded-md border border-fg/10 bg-surface-2/40 p-2.5">
          <TextField autoFocus placeholder="Topic (short title)" value={topic} onChange={(e) => setTopic(e.target.value)} />
          <textarea
            className="min-h-[64px] w-full resize-none rounded-md border border-fg/12 bg-surface px-2 py-1.5 text-[13px] text-fg outline-none placeholder:text-fg-subtle"
            placeholder="The finding (a class, pattern, API, decision…)"
            value={content}
            onChange={(e) => setContent(e.target.value)}
          />
          <div className="flex justify-end">
            <Button variant="primary" size="sm" onClick={() => void saveFinding()} disabled={!topic.trim() || !content.trim()}>
              Save finding
            </Button>
          </div>
        </div>
      )}

      <div className="mt-1 text-[10px] text-fg-subtle">
        {busy ? "Loading…" : `${items.length} finding${items.length === 1 ? "" : "s"}`}
      </div>
      <div className="mt-2 grid h-[55vh] grid-cols-[minmax(180px,40%)_1fr] gap-3">
        <div className="space-y-0.5 overflow-y-auto border-r border-fg/8 pr-2">
          {items.length === 0 && !busy && (
            <div className="px-2 py-3 text-[12px] text-fg-subtle">
              No findings recorded yet. The agent records them with the
              <span className="mono"> record_finding</span> tool as it explores the project.
            </div>
          )}
          {items.map((f) => (
            <button
              key={f.id}
              onClick={() => setSel(f)}
              className={`flex w-full flex-col gap-0.5 rounded px-2 py-1.5 text-left ${
                sel?.id === f.id ? "bg-accent/15" : "hover:bg-surface-2/60"
              }`}
            >
              <span className="truncate text-[12px] text-fg">{f.topic}</span>
              <span className="truncate text-[10px] text-fg-subtle">
                {f.finding_type} · {f.created_at}
              </span>
            </button>
          ))}
        </div>
        <div className="overflow-y-auto">
          {sel ? (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Badge>{sel.finding_type}</Badge>
                <span className="text-[10px] text-fg-subtle">
                  confidence {Math.round(sel.confidence * 100)}% · #{sel.id}
                </span>
              </div>
              <div className="text-sm font-semibold text-fg">{sel.topic}</div>
              <p className="whitespace-pre-wrap text-[13px] leading-relaxed text-fg-muted">
                {sel.content}
              </p>
            </div>
          ) : (
            <div className="grid h-full place-items-center text-[12px] text-fg-subtle">
              Select a finding.
            </div>
          )}
        </div>
      </div>
    </Dialog>
  );
}
