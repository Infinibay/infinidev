import { useAppState } from "@/state/store";
import { ActivityTab } from "./ActivityTab";
import { ChangesTab } from "./ChangesTab";
import { FilesTab } from "./FilesTab";
import { TerminalTab } from "./TerminalTab";

export type WorkbenchTab = "activity" | "changes" | "files" | "terminal";

interface Props {
  tab: WorkbenchTab;
  setTab: (t: WorkbenchTab) => void;
  selectedFile: string | null;
  openFile: (path: string) => void;
}

export function Workbench({ tab, setTab, selectedFile, openFile }: Props) {
  const s = useAppState();
  const tabs: { id: WorkbenchTab; label: string; badge?: number }[] = [
    { id: "activity", label: "Activity" },
    { id: "changes", label: "Changes", badge: s.changedFiles.length || undefined },
    { id: "files", label: "Files" },
    { id: "terminal", label: "Terminal" },
  ];

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 items-stretch border-b border-fg/8 bg-surface-1/40 text-xs">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex items-center gap-1.5 px-3.5 py-2.5 ${
              tab === t.id ? "border-b-2 border-accent text-fg" : "border-b-2 border-transparent text-fg-muted hover:text-fg"
            }`}
          >
            {t.label}
            {t.badge ? (
              <span className="rounded-full bg-accent/20 px-1.5 text-[10px] text-accent">{t.badge}</span>
            ) : null}
          </button>
        ))}
      </div>
      <div className="min-h-0 flex-1 overflow-hidden">
        {tab === "activity" && <ActivityTab />}
        {tab === "changes" && <ChangesTab openFile={openFile} />}
        {tab === "files" && <FilesTab selectedFile={selectedFile} openFile={openFile} />}
        {tab === "terminal" && <TerminalTab />}
      </div>
    </div>
  );
}
