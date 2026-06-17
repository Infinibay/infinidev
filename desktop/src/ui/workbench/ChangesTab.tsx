import { Badge } from "@harbor/components/display/Badge";
import { useAppState } from "@/state/store";

export function ChangesTab({ openFile }: { openFile: (path: string) => void }) {
  const s = useAppState();
  if (s.changedFiles.length === 0) {
    return (
      <div className="flex h-full items-center justify-center px-6 text-center text-sm text-fg-subtle">
        No changes yet. Files the agent creates or edits this session show up here for review.
      </div>
    );
  }
  return (
    <div className="h-full overflow-y-auto p-3">
      <ul className="space-y-1">
        {s.changedFiles.map((f) => (
          <li key={f.path}>
            <button
              onClick={() => openFile(f.path)}
              className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left hover:bg-surface-2/60"
            >
              <Badge tone={f.action === "create" ? "success" : "info"}>{f.action}</Badge>
              <span className="mono truncate text-xs text-fg">{f.path}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
