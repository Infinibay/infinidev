import { StatusDot } from "@harbor/components/display/StatusDot";
import { useEngine } from "@/state/store";
import { MenuBar } from "./MenuBar";

function basename(p: string): string {
  const parts = p.replace(/\/+$/, "").split("/");
  return parts[parts.length - 1] || p;
}

interface Props {
  onOpenSettings: () => void;
  onOpenSearch: () => void;
  onAction: (action: string) => void;
}

export function Topbar({ onOpenSettings, onOpenSearch, onAction }: Props) {
  const { state, transportKind } = useEngine();
  const tokens = state.usage?.total ?? 0;

  return (
    <header className="flex h-11 shrink-0 items-center gap-3 border-b border-fg/8 bg-surface-2/60 px-3">
      <span className="text-sm font-semibold text-accent">Infinidev</span>
      <MenuBar onAction={onAction} />
      {state.projectPath && (
        <span className="mono truncate text-[11px] text-fg-subtle" title={state.projectPath}>
          {basename(state.projectPath)}
        </span>
      )}
      <div className="ml-auto flex items-center gap-2 text-xs">
        <button
          onClick={onOpenSearch}
          className="rounded-md border border-fg/12 px-2 py-1 text-fg-muted hover:text-fg"
        >
          Search <span className="text-fg-subtle">⌘K</span>
        </button>
        <button
          onClick={onOpenSettings}
          title="Model & settings"
          className="mono rounded-md border border-fg/12 px-2 py-1 text-fg-muted hover:border-accent/40 hover:text-fg"
        >
          {state.model || "set model"}
        </button>
        {tokens > 0 && (
          <span className="rounded-md bg-surface px-2 py-1 text-[10px] text-fg-subtle">
            {tokens.toLocaleString()} tok
            {state.costUsd > 0 ? ` · $${state.costUsd.toFixed(4)}` : ""}
          </span>
        )}
        <button
          onClick={onOpenSettings}
          aria-label="Settings"
          title="Settings (⌘,)"
          className="flex h-7 w-7 items-center justify-center rounded-md text-[18px] leading-none text-fg-muted transition-colors hover:bg-surface-2 hover:text-fg"
        >
          ⚙
        </button>
        <StatusDot
          status={state.ready ? "online" : "degraded"}
          size={9}
          label={transportKind === "tauri" ? "engine" : "demo"}
        />
      </div>
    </header>
  );
}
