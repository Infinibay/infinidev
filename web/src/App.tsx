import { lazy, Suspense, useEffect, useState } from "react";
import { DropdownMenu, Tooltip } from "radix-ui";
import {
  ArrowUpRight,
  BookOpen,
  ChevronDown,
  ChevronRight,
  CircleGauge,
  Command,
  Ellipsis,
  FolderCode,
  GitCompareArrows,
  KeyRound,
  Menu,
  MessageSquare,
  Moon,
  PanelRightClose,
  PanelRightOpen,
  Plus,
  Search,
  Settings2,
  Sparkles,
  Sun,
  Terminal,
  Users,
  Wrench,
  X,
} from "lucide-react";
import { api, hasToken, setToken } from "./api";
import { Badge, ErrorNotice, Loading, Modal, Tip } from "./components";
import { useResource, useSession, useTeam } from "./hooks";
import type { Info, Models, SessionSummary, View } from "./types";
import { Workspace, Inspector } from "./Workspace";
const TeamView = lazy(() =>
  import("./TeamView").then((module) => ({ default: module.TeamView })),
);
const ProcessesView = lazy(() =>
  import("./ProcessesView").then((module) => ({
    default: module.ProcessesView,
  })),
);
const FilesView = lazy(() =>
  import("./FilesView").then((module) => ({ default: module.FilesView })),
);
const KnowledgeView = lazy(() =>
  import("./DataViews").then((module) => ({ default: module.KnowledgeView })),
);
const ToolsView = lazy(() =>
  import("./DataViews").then((module) => ({ default: module.ToolsView })),
);
const UsageView = lazy(() =>
  import("./DataViews").then((module) => ({ default: module.UsageView })),
);
const SettingsView = lazy(() =>
  import("./SettingsView").then((module) => ({ default: module.SettingsView })),
);

const navigation: {
  id: View;
  label: string;
  icon: typeof MessageSquare;
  shortcut?: string;
}[] = [
  { id: "workspace", label: "Workspace", icon: MessageSquare },
  { id: "team", label: "Team", icon: Users },
  { id: "processes", label: "Processes", icon: Terminal },
  { id: "changes", label: "Changes", icon: GitCompareArrows },
  { id: "knowledge", label: "Knowledge", icon: BookOpen },
  { id: "tools", label: "Tools", icon: Wrench },
  { id: "usage", label: "Usage", icon: CircleGauge },
  { id: "settings", label: "Settings", icon: Settings2 },
];

function Login({ onLogin }: { onLogin: () => void }) {
  const [value, setValue] = useState("");
  const [error, setError] = useState("");
  async function connect() {
    const entered = value.includes("#token=")
      ? new URLSearchParams(value.split("#")[1]).get("token") || ""
      : value.trim();
    setToken(entered);
    try {
      await api("/info");
      onLogin();
    } catch (e) {
      setError((e as Error).message);
    }
  }
  return (
    <div className="login-page">
      <div className="login-card">
        <div className="brand-mark">
          <Sparkles size={26} />
        </div>
        <div className="eyebrow">Infinidev web</div>
        <h1>
          Your workspace,
          <br />a clearer view.
        </h1>
        <p>
          Open the launch URL printed by <code>infinidev web</code>, or paste it
          below to connect to this local server.
        </p>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void connect();
          }}
        >
          <label>
            Launch URL or access token
            <input
              autoFocus
              type="password"
              autoComplete="off"
              value={value}
              onChange={(e) => setValue(e.target.value)}
              placeholder="Paste from your terminal"
            />
          </label>
          <ErrorNotice error={error} />
          <button className="button primary" disabled={!value.trim()}>
            <KeyRound size={16} />
            Connect to workspace
          </button>
        </form>
      </div>
    </div>
  );
}

export default function App() {
  const [authenticated, setAuthenticated] = useState(hasToken);
  if (!authenticated) return <Login onLogin={() => setAuthenticated(true)} />;
  return (
    <Tooltip.Provider delayDuration={300}>
      <Harness />
    </Tooltip.Provider>
  );
}

function Harness() {
  const [view, setView] = useState<View>("workspace");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [inspector, setInspector] = useState(true);
  const [mobileNav, setMobileNav] = useState(false);
  const [palette, setPalette] = useState(false);
  const [query, setQuery] = useState("");
  const [sessionSearch, setSessionSearch] = useState("");
  const [error, setError] = useState("");
  const [creating, setCreating] = useState(false);
  const [fileDirty, setFileDirty] = useState(false);
  const [rename, setRename] = useState<SessionSummary | null>(null);
  const [title, setTitle] = useState("");
  const [theme, setTheme] = useState(
    localStorage.getItem("infinidev-theme") || "light",
  );
  const info = useResource<Info>("/info", 10000);
  const sessions = useResource<{ sessions: SessionSummary[] }>(
    "/sessions",
    3000,
  );
  const models = useResource<Models>("/models");
  const live = useSession(sessionId);
  const teamState = useTeam(sessionId);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("infinidev-theme", theme);
  }, [theme]);
  useEffect(() => {
    if (sessionId || !sessions.data) return;
    const saved = localStorage.getItem(
      `infinidev-session:${info.data?.cwd || ""}`,
    );
    const first =
      sessions.data.sessions.find((s) => s.session_id === saved) ||
      sessions.data.sessions[0];
    if (first) setSessionId(first.session_id);
    else if (!creating) void newSession();
  }, [sessions.data, info.data]);
  useEffect(() => {
    if (sessionId && info.data)
      localStorage.setItem(`infinidev-session:${info.data.cwd}`, sessionId);
  }, [sessionId, info.data]);
  useEffect(() => {
    const listener = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setPalette((open) => !open);
      }
      if (event.key === "Escape") setMobileNav(false);
    };
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, []);
  async function newSession() {
    if (creating) return;
    if (fileDirty && !window.confirm("Discard your unsaved file edits?"))
      return;
    setCreating(true);
    setError("");
    try {
      const created = await api<SessionSummary>("/sessions", {});
      setSessionId(created.session_id);
      setView("workspace");
      sessions.reload();
      setMobileNav(false);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setCreating(false);
    }
  }
  function navigate(next: View) {
    if (
      next !== view &&
      fileDirty &&
      !window.confirm("Discard your unsaved file edits?")
    )
      return;
    setView(next);
    setMobileNav(false);
    setPalette(false);
  }
  function select(id: string) {
    setSessionId(id);
    navigate("workspace");
  }
  function draft(value: string) {
    if (sessionId) setDrafts((values) => ({ ...values, [sessionId]: value }));
  }
  function ask(text: string) {
    draft(text);
    navigate("workspace");
  }
  const current =
    live.session ||
    sessions.data?.sessions.find((s) => s.session_id === sessionId);
  async function saveTitle() {
    if (!rename) return;
    try {
      await api(`/sessions/${rename.session_id}`, { title }, "PATCH");
      setRename(null);
      sessions.reload();
    } catch (e) {
      setError((e as Error).message);
    }
  }
  return (
    <div className={`harness ${mobileNav ? "nav-open" : ""}`}>
      {mobileNav && (
        <button
          className="nav-backdrop"
          aria-label="Close navigation"
          onClick={() => setMobileNav(false)}
        />
      )}
      <aside className="sidebar">
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            navigate("workspace");
          }}
        >
          <span className="brand-mark">
            <Sparkles size={23} strokeWidth={1.7} />
          </span>
          <span>
            infinidev<span className="brand-beta">WEB</span>
          </span>
        </a>
        <div className="project-card">
          <span className="project-icon">
            <FolderCode size={18} />
          </span>
          <div>
            <strong>{info.data?.workspace || "Workspace"}</strong>
            <span>Local workspace</span>
          </div>
          <span className="project-connected" />
        </div>
        <button
          className="command-button"
          onClick={() => {
            setQuery("");
            setPalette(true);
          }}
        >
          <Search size={15} />
          <span>Jump to…</span>
          <kbd>⌘ K</kbd>
        </button>
        <div className="nav-label">WORKSPACE</div>
        <nav aria-label="Main navigation">
          {navigation.slice(0, 6).map((item) => (
            <button
              className={`nav-item ${view === item.id ? "active" : ""}`}
              key={item.id}
              onClick={() => navigate(item.id)}
            >
              <item.icon size={18} strokeWidth={1.7} />
              <span>{item.label}</span>
              {item.id === "team" &&
                Object.keys(teamState.team?.board.agents || {}).length > 0 && (
                  <span className="nav-count">
                    {Object.keys(teamState.team?.board.agents || {}).length}
                  </span>
                )}
            </button>
          ))}
        </nav>
        <div className="sessions-heading">
          <span className="nav-label">SESSIONS</span>
          <Tip label="New session">
            <button
              disabled={creating}
              className="icon-button"
              aria-label="New session"
              onClick={() => void newSession()}
            >
              <Plus size={16} />
            </button>
          </Tip>
        </div>
        <label className="session-search">
          <Search size={13} />
          <input
            aria-label="Search sessions"
            placeholder="Search sessions"
            value={sessionSearch}
            onChange={(e) => setSessionSearch(e.target.value)}
          />
        </label>
        <div className="session-list">
          {sessions.data?.sessions
            .filter((s) =>
              s.title.toLowerCase().includes(sessionSearch.toLowerCase()),
            )
            .map((session) => (
              <div
                className={`session-item ${session.session_id === sessionId ? "active" : ""}`}
                key={session.session_id}
              >
                <button onClick={() => select(session.session_id)}>
                  <span
                    className={
                      ["running", "waiting", "queued"].includes(session.status)
                        ? "pulse-dot"
                        : "session-dot"
                    }
                  />
                  <span>{session.title}</span>
                </button>
                <DropdownMenu.Root>
                  <DropdownMenu.Trigger asChild>
                    <button
                      className="session-menu"
                      aria-label={`Options for ${session.title}`}
                    >
                      <Ellipsis size={14} />
                    </button>
                  </DropdownMenu.Trigger>
                  <DropdownMenu.Portal>
                    <DropdownMenu.Content className="dropdown" sideOffset={5}>
                      <DropdownMenu.Item
                        onSelect={() => {
                          setRename(session);
                          setTitle(session.title);
                        }}
                      >
                        Rename session
                      </DropdownMenu.Item>
                    </DropdownMenu.Content>
                  </DropdownMenu.Portal>
                </DropdownMenu.Root>
              </div>
            ))}
        </div>
        <div className="sidebar-bottom">
          <nav aria-label="Configuration navigation">
            {navigation.slice(6).map((item) => (
              <button
                key={item.id}
                className={`nav-item ${view === item.id ? "active" : ""}`}
                onClick={() => navigate(item.id)}
              >
                <item.icon size={17} />
                <span>{item.label}</span>
              </button>
            ))}
          </nav>
          <div className="sidebar-footer">
            <span>
              <span
                className={`connection-dot ${info.error ? "disconnected" : ""}`}
              />
              {info.error ? "Disconnected" : "Local server"}
            </span>
            <Tip
              label={theme === "light" ? "Use dark theme" : "Use light theme"}
            >
              <button
                className="icon-button"
                onClick={() => setTheme(theme === "light" ? "dark" : "light")}
                aria-label="Toggle theme"
              >
                {theme === "light" ? <Moon size={15} /> : <Sun size={15} />}
              </button>
            </Tip>
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <button
            className="icon-button mobile-menu"
            onClick={() => setMobileNav(true)}
            aria-label="Open navigation"
          >
            <Menu size={19} />
          </button>
          <div className="breadcrumbs">
            <FolderCode size={15} />
            <span>{info.data?.workspace || "Workspace"}</span>
            <ChevronRight size={13} />
            <strong>
              {view === "workspace"
                ? current?.title || "New session"
                : navigation.find((item) => item.id === view)?.label}
            </strong>
          </div>
          <div className="topbar-actions">
            <button className="model-pill" onClick={() => navigate("settings")}>
              <span className="model-dot" />
              {models.data?.current.split("/").pop() || "Model"}
              <ChevronDown size={12} />
            </button>
            {view === "workspace" && (
              <Tip
                label={
                  inspector ? "Hide session overview" : "Show session overview"
                }
              >
                <button
                  className="icon-button"
                  aria-label="Toggle session overview"
                  onClick={() => setInspector((v) => !v)}
                >
                  {inspector ? (
                    <PanelRightClose size={18} />
                  ) : (
                    <PanelRightOpen size={18} />
                  )}
                </button>
              </Tip>
            )}
          </div>
        </header>
        {(error || info.error || sessions.error) && (
          <ErrorNotice
            error={error || info.error || sessions.error}
            retry={() => {
              setError("");
              info.reload();
              sessions.reload();
            }}
          />
        )}
        <main className="main-content">
          <Suspense fallback={<Loading />}>
            {view === "workspace" ? (
              <>
                <Workspace
                  key={sessionId}
                  session={live.session}
                  sessionId={sessionId}
                  connection={live.connection}
                  models={models.data}
                  draft={drafts[sessionId || ""] || ""}
                  setDraft={draft}
                  onSent={sessions.reload}
                  creating={creating}
                />
                {inspector && (
                  <Inspector
                    session={live.session}
                    team={teamState.team}
                    onView={navigate}
                  />
                )}
              </>
            ) : view === "team" ? (
              <TeamView
                team={teamState.team}
                sessionId={sessionId}
                error={teamState.error}
                ask={ask}
              />
            ) : view === "processes" ? (
              <ProcessesView />
            ) : view === "changes" ? (
              <FilesView onDirty={setFileDirty} />
            ) : view === "knowledge" ? (
              <KnowledgeView />
            ) : view === "tools" ? (
              <ToolsView />
            ) : view === "usage" ? (
              <UsageView />
            ) : (
              <SettingsView
                onSaved={() => {
                  models.reload();
                  info.reload();
                }}
              />
            )}
          </Suspense>
        </main>
        <footer className="statusbar">
          <span>
            <span className="status-dot" />
            {current?.status && current.status !== "idle"
              ? current.status
              : "Ready"}
            <span className="statusbar-divider" />
            {info.data?.cwd || "Connecting…"}
          </span>
          <span>
            infinidev {info.data?.version ? `v${info.data.version}` : ""}
            <span className="statusbar-divider" />
            {info.data?.provider}
          </span>
        </footer>
      </div>
      <Modal
        open={palette}
        onOpenChange={setPalette}
        title="Jump to anything"
        description="Navigate the workspace or open a session."
      >
        <label className="palette-search">
          <Search size={18} />
          <input
            autoFocus
            aria-label="Search commands and sessions"
            placeholder="Search views and sessions…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </label>
        <div className="palette-results">
          <button
            onClick={() => {
              setPalette(false);
              void newSession();
            }}
          >
            <Plus size={18} />
            <span>New session</span>
            <ArrowUpRight size={14} />
          </button>
          {navigation
            .filter((item) =>
              item.label.toLowerCase().includes(query.toLowerCase()),
            )
            .map((item) => (
              <button key={item.id} onClick={() => navigate(item.id)}>
                <item.icon size={18} />
                <span>{item.label}</span>
                <ChevronRight size={14} />
              </button>
            ))}
          {sessions.data?.sessions
            .filter((s) => s.title.toLowerCase().includes(query.toLowerCase()))
            .slice(0, 8)
            .map((session) => (
              <button
                key={session.session_id}
                onClick={() => select(session.session_id)}
              >
                <MessageSquare size={17} />
                <span>{session.title}</span>
                <Badge status={session.status} />
              </button>
            ))}
        </div>
        <div className="palette-footer">
          <Command size={12} />K to open <span>·</span> Esc to close
        </div>
      </Modal>
      <Modal
        open={!!rename}
        onOpenChange={(open) => {
          if (!open) setRename(null);
        }}
        title="Rename session"
      >
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void saveTitle();
          }}
        >
          <label className="form-field">
            Session name
            <input
              autoFocus
              value={title}
              maxLength={80}
              onChange={(e) => setTitle(e.target.value)}
            />
          </label>
          <div className="question-actions">
            <button className="button primary" disabled={!title.trim()}>
              Save name
            </button>
          </div>
        </form>
      </Modal>
    </div>
  );
}
