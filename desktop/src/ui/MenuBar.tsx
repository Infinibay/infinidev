import { Menu, MenuItem, MenuSeparator } from "@harbor/components/overlays/Menu";
import { useAppState } from "@/state/store";

/** A single top-level menu (File, Edit, …) rendered with Harbor's `Menu`.
 *  The trigger is a flat text button; the popover holds `MenuItem`s. */
function TopMenu({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <Menu
      trigger={
        <button
          type="button"
          className="rounded px-2 py-0.5 text-xs text-fg-muted outline-none transition-colors hover:bg-surface-2 hover:text-fg aria-expanded:bg-surface-2 aria-expanded:text-fg"
        >
          {label}
        </button>
      }
    >
      {children}
    </Menu>
  );
}

/** In-window application menu bar (works in the browser demo *and* the native
 *  build, unlike the OS menu which only exists under Tauri). Every item routes
 *  through `onAction`, the same dispatch the slash commands and the native menu
 *  use, so there is one source of truth for what each command does. */
export function MenuBar({ onAction }: { onAction: (action: string) => void }) {
  const s = useAppState();
  return (
    <nav className="flex items-center gap-0.5" aria-label="Application menu">
      <TopMenu label="File">
        <MenuItem icon="📂" shortcut="⌘O" onClick={() => onAction("open_folder")}>
          Open Folder…
        </MenuItem>
        <MenuItem icon="✦" shortcut="⌘N" onClick={() => onAction("new_session")}>
          New Session
        </MenuItem>
        <MenuItem icon="⬇" onClick={() => onAction("export")}>
          Export Conversation…
        </MenuItem>
        <MenuSeparator />
        <MenuItem icon="⚙" shortcut="⌘," onClick={() => onAction("settings")}>
          Settings…
        </MenuItem>
      </TopMenu>

      <TopMenu label="Edit">
        <MenuItem icon="⌕" shortcut="⌘K" onClick={() => onAction("search")}>
          Find in Project…
        </MenuItem>
        <MenuSeparator />
        <MenuItem icon="🗑" danger onClick={() => onAction("clear")}>
          Clear Conversation
        </MenuItem>
      </TopMenu>

      <TopMenu label="View">
        <MenuItem onClick={() => onAction("show_activity")}>Activity</MenuItem>
        <MenuItem onClick={() => onAction("show_changes")}>Changes</MenuItem>
        <MenuItem onClick={() => onAction("show_files")}>Files</MenuItem>
        <MenuItem onClick={() => onAction("show_terminal")}>Terminal</MenuItem>
        <MenuSeparator />
        <MenuItem onClick={() => onAction("show_knowledge")}>Knowledge</MenuItem>
        <MenuItem onClick={() => onAction("show_notes")}>Notes</MenuItem>
        <MenuItem onClick={() => onAction("show_background")}>Background Tasks</MenuItem>
        <MenuItem onClick={() => onAction("show_docs")}>Documentation</MenuItem>
        <MenuItem onClick={() => onAction("show_debug")}>Debug…</MenuItem>
      </TopMenu>

      <TopMenu label="Agent">
        <MenuItem disabled={!s.busy} danger shortcut="⌘." onClick={() => onAction("stop")}>
          Stop Turn
        </MenuItem>
        <MenuItem onClick={() => onAction("models")}>Choose Model…</MenuItem>
      </TopMenu>

      <TopMenu label="Help">
        <MenuItem onClick={() => onAction("about")}>About Infinidev</MenuItem>
      </TopMenu>
    </nav>
  );
}
