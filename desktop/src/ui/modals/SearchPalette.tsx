import { useEffect, useRef, useState } from "react";
import { Dialog } from "@harbor/components/overlays/Dialog";
import { TextField } from "@harbor/components/inputs/TextField";
import { useEngine } from "@/state/store";
import type { RsSearchHit } from "@/api/engineEvent";

interface Props {
  open: boolean;
  onClose: () => void;
  openFile: (path: string) => void;
}

export function SearchPalette({ open, onClose, openFile }: Props) {
  const { commands } = useEngine();
  const [q, setQ] = useState("");
  const [results, setResults] = useState<RsSearchHit[]>([]);
  const [busy, setBusy] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!open) {
      setQ("");
      setResults([]);
    }
  }, [open]);

  useEffect(() => {
    if (timer.current) clearTimeout(timer.current);
    if (!q.trim()) {
      setResults([]);
      return;
    }
    timer.current = setTimeout(() => {
      setBusy(true);
      commands
        .search(q.trim())
        .then(setResults)
        .catch(() => setResults([]))
        .finally(() => setBusy(false));
    }, 250);
  }, [q, commands]);

  return (
    <Dialog open={open} onClose={onClose} size="lg" title="Search in project">
      <TextField autoFocus placeholder="Search file contents…" value={q} onChange={(e) => setQ(e.target.value)} />
      <div className="mt-1 text-[10px] text-fg-subtle">{busy ? "Searching…" : `${results.length} matches`}</div>
      <div className="mt-2 max-h-[55vh] space-y-0.5 overflow-y-auto">
        {results.map((r, i) => (
          <button
            key={`${r.path}:${r.line}:${i}`}
            onClick={() => {
              openFile(r.path);
              onClose();
            }}
            className="flex w-full items-baseline gap-2 rounded px-2 py-1 text-left hover:bg-surface-2/60"
          >
            <span className="mono shrink-0 text-[11px] text-accent">
              {r.path}:{r.line}
            </span>
            <span className="mono truncate text-[11px] text-fg-muted">{r.text}</span>
          </button>
        ))}
      </div>
    </Dialog>
  );
}
