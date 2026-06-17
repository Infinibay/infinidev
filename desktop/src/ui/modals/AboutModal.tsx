import { Dialog } from "@harbor/components/overlays/Dialog";
import { useEngine } from "@/state/store";

interface Props {
  open: boolean;
  onClose: () => void;
}

/** "About Infinidev" — app identity plus the live backend/model the session is
 *  wired to, so it doubles as a quick status panel. */
export function AboutModal({ open, onClose }: Props) {
  const { state, transportKind } = useEngine();
  return (
    <Dialog open={open} onClose={onClose} size="sm" title="About Infinidev">
      <div className="space-y-3 text-sm">
        <div>
          <div className="text-base font-semibold text-accent">Infinidev</div>
          <div className="text-fg-muted">Autonomous coding agent — local, embedded Rust engine.</div>
        </div>
        <dl className="space-y-1 text-[12px]">
          <Row label="Backend" value={transportKind === "tauri" ? "Embedded engine (Tauri)" : "Demo (browser)"} />
          <Row label="Model" value={state.model || "—"} />
          <Row label="Project" value={state.projectPath || "—"} mono />
        </dl>
        <div className="text-[11px] text-fg-subtle">© 2026 Infinibay · AGPL-3.0-or-later</div>
      </div>
    </Dialog>
  );
}

function Row({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-baseline gap-2">
      <dt className="w-20 shrink-0 text-fg-subtle">{label}</dt>
      <dd className={`min-w-0 truncate text-fg-muted${mono ? " mono text-[11px]" : ""}`} title={value}>
        {value}
      </dd>
    </div>
  );
}
