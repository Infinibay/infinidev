import { Conversation } from "./conversation/Conversation";
import { FileEditor, basename } from "./FileEditor";
import { useEngine } from "@/state/store";

interface Props {
  openFiles: string[];
  /** "chat" or an open file path. */
  active: string;
  setActive: (key: string) => void;
  closeFile: (path: string) => void;
  dirty: Record<string, boolean>;
  onDirty: (path: string, dirty: boolean) => void;
  onCommand: (action: string) => void;
}

/** The main left pane: a tab strip with a permanent "Chat" tab plus one tab per
 *  open file. Chat and every file editor stay mounted (inactive ones hidden) so
 *  conversation scroll and unsaved edits survive switching tabs. */
export function MainPane({ openFiles, active, setActive, closeFile, dirty, onDirty, onCommand }: Props) {
  const { commands } = useEngine();
  const hasFiles = openFiles.length > 0;

  return (
    <div className="flex h-full flex-col">
      {hasFiles && (
        <div className="flex shrink-0 items-stretch overflow-x-auto border-b border-fg/10 bg-surface-2/40">
          <TabButton active={active === "chat"} onClick={() => setActive("chat")}>
            <span className="text-accent">◆</span>
            <span>Chat</span>
          </TabButton>
          {openFiles.map((p) => (
            <TabButton key={p} active={active === p} onClick={() => setActive(p)} title={p}>
              <span className="mono max-w-[160px] truncate">{basename(p)}</span>
              <span className="relative grid h-3 w-3 shrink-0 place-items-center">
                {dirty[p] && <span className="text-warning group-hover:opacity-0">●</span>}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    closeFile(p);
                  }}
                  className="absolute inset-0 grid place-items-center opacity-0 transition-opacity hover:text-fg group-hover:opacity-100"
                  aria-label={`Close ${basename(p)}`}
                >
                  ×
                </button>
              </span>
            </TabButton>
          ))}
        </div>
      )}
      <div className="relative min-h-0 flex-1">
        <div className={`absolute inset-0 ${active === "chat" ? "" : "hidden"}`}>
          <Conversation onCommand={onCommand} />
        </div>
        {openFiles.map((p) => (
          <div key={p} className={`absolute inset-0 ${active === p ? "" : "hidden"}`}>
            <FileEditor path={p} commands={commands} visible={active === p} onDirty={onDirty} />
          </div>
        ))}
      </div>
    </div>
  );
}

function TabButton({
  active,
  onClick,
  title,
  children,
}: {
  active: boolean;
  onClick: () => void;
  title?: string;
  children: React.ReactNode;
}) {
  return (
    <div
      onClick={onClick}
      title={title}
      className={`group flex cursor-pointer items-center gap-1.5 border-r border-fg/8 px-3 py-1.5 text-[11px] ${
        active ? "bg-surface text-fg" : "text-fg-muted hover:bg-surface-2/50"
      }`}
    >
      {children}
    </div>
  );
}
