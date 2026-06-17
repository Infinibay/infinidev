import { useCallback, useEffect, useState } from "react";
import { useToast } from "@harbor/components/feedback/Toast";
import { useEngine } from "@/state/store";
import type { Commands } from "@/api/transport";
import type { RsTreeEntry } from "@/api/engineEvent";

interface Props {
  /** The path open in the main pane, so the tree can highlight it. */
  selectedFile: string | null;
  /** Open a file as a tab in the main pane. */
  openFile: (path: string) => void;
}

const parentOf = (p: string) => p.split("/").slice(0, -1).join("/");

/** File explorer (tree) for the workbench, with create / rename / delete. Files
 *  open as tabs in the main pane; editing happens there, not here. */
export function FilesTab({ selectedFile, openFile }: Props) {
  const { commands, state } = useEngine();
  const { push } = useToast();
  const [root, setRoot] = useState<RsTreeEntry[]>([]);
  const [err, setErr] = useState<string | null>(null);
  // Bumped after a filesystem mutation; open folders re-fetch their children.
  const [version, setVersion] = useState(0);
  const refresh = useCallback(() => setVersion((v) => v + 1), []);

  useEffect(() => {
    setErr(null);
    commands.fsTree("").then(setRoot).catch((e) => setErr(String(e)));
  }, [commands, state.projectNonce, version]);

  const create = useCallback(
    async (parent: string, dir: boolean) => {
      const name = window.prompt(dir ? "New folder name" : "New file name")?.trim();
      if (!name) return;
      const path = parent ? `${parent}/${name}` : name;
      try {
        await commands.fsCreate(path, dir);
        refresh();
        if (!dir) openFile(path);
      } catch (e) {
        push({ title: "Could not create", description: String((e as Error).message), tone: "danger" });
      }
    },
    [commands, openFile, push, refresh],
  );

  const rename = useCallback(
    async (entry: RsTreeEntry) => {
      const name = window.prompt("Rename to", entry.name)?.trim();
      if (!name || name === entry.name) return;
      const parent = parentOf(entry.path);
      const to = parent ? `${parent}/${name}` : name;
      try {
        await commands.fsRename(entry.path, to);
        refresh();
      } catch (e) {
        push({ title: "Rename failed", description: String((e as Error).message), tone: "danger" });
      }
    },
    [commands, push, refresh],
  );

  const remove = useCallback(
    async (entry: RsTreeEntry) => {
      const what = entry.is_dir ? "folder and everything in it" : "file";
      if (!window.confirm(`Delete ${entry.name}? This ${what} cannot be recovered.`)) return;
      try {
        await commands.fsDelete(entry.path);
        refresh();
      } catch (e) {
        push({ title: "Delete failed", description: String((e as Error).message), tone: "danger" });
      }
    },
    [commands, push, refresh],
  );

  return (
    <div className="flex h-full flex-col">
      <div className="flex shrink-0 items-center gap-1 border-b border-fg/8 px-2 py-1">
        <span className="mr-auto text-[10px] uppercase tracking-wider text-fg-subtle">Explorer</span>
        <button onClick={() => void create("", false)} title="New file" className="rounded px-1.5 py-0.5 text-xs text-fg-muted hover:bg-surface-2/60 hover:text-fg">
          ＋
        </button>
        <button onClick={() => void create("", true)} title="New folder" className="rounded px-1.5 py-0.5 text-xs text-fg-muted hover:bg-surface-2/60 hover:text-fg">
          ＋📁
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto py-1">
        {err && <div className="px-3 py-2 text-xs text-danger">{err}</div>}
        {root.length === 0 && !err && <div className="px-3 py-2 text-xs text-fg-subtle">Empty folder.</div>}
        {root.map((e) => (
          <TreeNode
            key={e.path}
            entry={e}
            depth={0}
            commands={commands}
            openFile={openFile}
            selected={selectedFile}
            version={version}
            onCreate={create}
            onRename={rename}
            onDelete={remove}
          />
        ))}
      </div>
    </div>
  );
}

interface NodeProps {
  entry: RsTreeEntry;
  depth: number;
  commands: Commands;
  openFile: (path: string) => void;
  selected: string | null;
  version: number;
  onCreate: (parent: string, dir: boolean) => void;
  onRename: (entry: RsTreeEntry) => void;
  onDelete: (entry: RsTreeEntry) => void;
}

function TreeNode({ entry, depth, commands, openFile, selected, version, onCreate, onRename, onDelete }: NodeProps) {
  const [open, setOpen] = useState(false);
  const [children, setChildren] = useState<RsTreeEntry[] | null>(null);

  // Re-fetch an open folder's children when the tree version bumps (after a
  // create/rename/delete somewhere), so the view stays in sync without collapsing.
  useEffect(() => {
    if (open) commands.fsTree(entry.path).then(setChildren).catch(() => setChildren([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version]);

  const toggle = useCallback(() => {
    if (entry.is_dir) {
      const next = !open;
      setOpen(next);
      if (next && children == null) {
        commands.fsTree(entry.path).then(setChildren).catch(() => setChildren([]));
      }
    } else {
      openFile(entry.path);
    }
  }, [entry, open, children, commands, openFile]);

  const sel = selected === entry.path;
  return (
    <div>
      <div
        className={`group flex items-center gap-1 pr-1 text-xs ${
          sel ? "bg-accent/15 text-fg" : "text-fg-muted hover:bg-surface-2/50"
        }`}
      >
        <button onClick={toggle} style={{ paddingLeft: 8 + depth * 12 }} className="flex min-w-0 flex-1 items-center gap-1 py-0.5 text-left">
          <span className="w-3 text-center text-fg-subtle">{entry.is_dir ? (open ? "▾" : "▸") : ""}</span>
          <span className="truncate">{entry.name}</span>
        </button>
        <div className="flex shrink-0 items-center opacity-0 transition-opacity group-hover:opacity-100">
          {entry.is_dir && (
            <button onClick={() => onCreate(entry.path, false)} title="New file here" className="px-1 text-fg-subtle hover:text-fg">
              ＋
            </button>
          )}
          <button onClick={() => onRename(entry)} title="Rename" className="px-1 text-fg-subtle hover:text-fg">
            ✎
          </button>
          <button onClick={() => onDelete(entry)} title="Delete" className="px-1 text-fg-subtle hover:text-danger">
            🗑
          </button>
        </div>
      </div>
      {open &&
        children?.map((c) => (
          <TreeNode
            key={c.path}
            entry={c}
            depth={depth + 1}
            commands={commands}
            openFile={openFile}
            selected={selected}
            version={version}
            onCreate={onCreate}
            onRename={onRename}
            onDelete={onDelete}
          />
        ))}
    </div>
  );
}
