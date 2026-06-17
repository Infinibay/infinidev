import { useEffect, useState } from "react";
import { Dialog } from "@harbor/components/overlays/Dialog";
import { Button } from "@harbor/components/buttons/Button";
import { useToast } from "@harbor/components/feedback/Toast";
import { useEngine } from "@/state/store";
import type { NoteRow } from "@/api/engineEvent";

interface Props {
  open: boolean;
  onClose: () => void;
}

/** Browser for the agent's working-memory notes (`record_note`). Mirrors the
 *  TUI's notes browser: a scrollable list of what the agent has jotted, which
 *  the engine re-injects into context at the start of every turn. */
export function NotesModal({ open, onClose }: Props) {
  const { commands, state } = useEngine();
  const { push } = useToast();
  const [notes, setNotes] = useState<NoteRow[]>([]);
  const [busy, setBusy] = useState(false);

  const reload = () => {
    setBusy(true);
    commands
      .notesList()
      .then(setNotes)
      .catch(() => setNotes([]))
      .finally(() => setBusy(false));
  };

  useEffect(() => {
    if (!open) return;
    reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, state.projectNonce]);

  const clearAll = async () => {
    if (!notes.length) return;
    if (!window.confirm("Clear all notes? The agent loses this working memory.")) return;
    try {
      await commands.notesClear();
      setNotes([]);
      push({ title: "Notes cleared", tone: "success" });
    } catch (e) {
      push({ title: "Could not clear", description: String((e as Error).message), tone: "danger" });
    }
  };

  return (
    <Dialog open={open} onClose={onClose} size="md" title="Notes — agent working memory">
      <div className="flex items-center justify-between">
        <div className="text-[10px] text-fg-subtle">
          {busy ? "Loading…" : `${notes.length} note${notes.length === 1 ? "" : "s"}`} · re-injected each turn
        </div>
        <Button variant="destructive" size="sm" disabled={!notes.length} onClick={() => void clearAll()}>
          Clear all
        </Button>
      </div>
      <div className="mt-2 max-h-[55vh] space-y-1.5 overflow-y-auto pr-1">
        {notes.length === 0 && !busy && (
          <div className="px-2 py-6 text-center text-[12px] text-fg-subtle">
            No notes yet. The agent jots them with the
            <span className="mono"> record_note</span> tool; they persist across turns
            and steer its later work.
          </div>
        )}
        {notes.map((n, i) => (
          <div key={n.id} className="flex gap-2 rounded-md border border-fg/8 bg-surface-2/40 px-2.5 py-1.5">
            <span className="mono shrink-0 text-[11px] text-fg-subtle">{i + 1}.</span>
            <div className="min-w-0">
              <p className="whitespace-pre-wrap text-[13px] leading-relaxed text-fg">{n.note}</p>
              <span className="text-[10px] text-fg-subtle">{n.created_at}</span>
            </div>
          </div>
        ))}
      </div>
    </Dialog>
  );
}
