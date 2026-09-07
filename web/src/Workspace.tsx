import { useRef, useState } from "react";
import {
  ArrowDown,
  ArrowUp,
  Bot,
  Check,
  ChevronDown,
  Copy,
  CornerDownLeft,
  FileSearch,
  GitBranch,
  LoaderCircle,
  MessageSquare,
  ShieldCheck,
  Sparkles,
  Square,
  Terminal,
  Users,
} from "lucide-react";
import { api } from "./api";
import {
  Avatar,
  Badge,
  ErrorNotice,
  JsonDetails,
  Markdown,
} from "./components";
import { useFollow } from "./hooks";
import type { Message, Models, Pending, Session, Team, View } from "./types";

function MessageCard({ message }: { message: Message }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    await navigator.clipboard.writeText(message.text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  if (message.kind === "status")
    return (
      <div className="status-message">
        <span className="status-dot" />
        {message.text}
      </div>
    );
  if (
    message.kind === "tool" ||
    message.kind === "reasoning" ||
    message.kind === "thinking"
  ) {
    return (
      <details
        className={`activity-message ${message.state === "error" ? "activity-error" : ""}`}
      >
        <summary>
          {message.state === "running" || message.streaming ? (
            <LoaderCircle size={15} className="spin" />
          ) : message.kind === "tool" ? (
            <Terminal size={15} />
          ) : (
            <Sparkles size={15} />
          )}
          <span>{message.kind === "tool" ? message.text : "Reasoning"}</span>
          <small>{message.speaker}</small>
          {message.state && (
            <span className="activity-result">
              {message.state === "completed" ? (
                <Check size={13} />
              ) : (
                message.state
              )}
            </span>
          )}
          <ChevronDown size={14} />
        </summary>
        <div className="activity-body">
          {message.kind === "tool" ? (
            <>
              {message.data?.tool_arguments != null && (
                <JsonDetails
                  label="Arguments"
                  value={message.data.tool_arguments}
                />
              )}
              <pre>
                {String(
                  message.data?.tool_result_full ??
                    message.data?.tool_detail ??
                    "Waiting for output…",
                )}
              </pre>
              {message.data?.tool_error != null &&
                Boolean(message.data.tool_error) && (
                  <pre className="error-text">
                    {String(message.data.tool_error)}
                  </pre>
                )}
            </>
          ) : (
            <Markdown text={message.text} />
          )}
        </div>
      </details>
    );
  }
  const user =
    message.kind === "user" ||
    message.speaker.toLowerCase() === "user" ||
    message.speaker === "You";
  return (
    <article
      className={`message ${user ? "message-user" : ""} ${message.kind === "error" ? "message-error" : ""}`}
    >
      <div className="message-avatar">
        {user ? (
          <Avatar name="You" small />
        ) : (
          <span className="agent-mark">
            <Sparkles size={17} />
          </span>
        )}
      </div>
      <div className="message-main">
        <div className="message-meta">
          <strong>{user ? "You" : message.speaker}</strong>
          {!user && (
            <span>
              {" "}
              {message.kind === "error" ? "Execution error" : "Agent"}
            </span>
          )}
          {message.created_at && (
            <time>
              {new Date(message.created_at * 1000).toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              })}
            </time>
          )}
          <button
            className="icon-button copy-button"
            onClick={() => void copy()}
            aria-label="Copy message"
          >
            {copied ? <Check size={13} /> : <Copy size={13} />}
          </button>
        </div>
        <Markdown text={message.text} />
        {message.streaming && <span className="typing-cursor" />}
        {message.traceback && (
          <JsonDetails label="Error details" value={message.traceback} />
        )}
      </div>
    </article>
  );
}

function Question({
  pending,
  sessionId,
}: {
  pending: Pending;
  sessionId: string;
}) {
  const [answer, setAnswer] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function respond(text: string | null) {
    setBusy(true);
    setError("");
    try {
      await api(`/sessions/${sessionId}/answers`, {
        request_id: pending.request_id,
        text,
      });
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  }
  return (
    <section className="question-card">
      <div className="question-title">
        <ShieldCheck size={18} />
        <strong>
          {pending.kind === "permission"
            ? "Your approval is needed"
            : "A question for you"}
        </strong>
      </div>
      <Markdown text={pending.prompt} />
      {pending.details && <pre>{pending.details}</pre>}
      <ErrorNotice error={error} />
      {pending.kind === "permission" ? (
        <div className="question-actions">
          <button
            className="button"
            disabled={busy}
            onClick={() => void respond("deny")}
          >
            Deny
          </button>
          <button
            className="button primary"
            disabled={busy}
            onClick={() => void respond("allow")}
          >
            Allow this action
          </button>
        </div>
      ) : (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (answer.trim()) void respond(answer);
          }}
        >
          <textarea
            aria-label="Answer the agent"
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
            placeholder="Write your answer…"
          />
          <div className="question-actions">
            <button
              type="button"
              className="button"
              disabled={busy}
              onClick={() => void respond(null)}
            >
              Skip
            </button>
            <button
              className="button primary"
              disabled={busy || !answer.trim()}
            >
              Send answer
            </button>
          </div>
        </form>
      )}
    </section>
  );
}

export function Workspace({
  session,
  sessionId,
  connection,
  models,
  draft,
  setDraft,
  onSent,
  creating,
}: {
  session: Session | null;
  sessionId: string | null;
  connection: string;
  models: Models | null;
  draft: string;
  setDraft: (value: string) => void;
  onSent: () => void;
  creating: boolean;
}) {
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");
  const textarea = useRef<HTMLTextAreaElement>(null);
  const request = useRef<{ text: string; id: string } | null>(null);
  const follow = useFollow(session?.messages);
  const busy =
    !!session &&
    ["running", "queued", "waiting", "cancelling"].includes(session.status);
  const send = async () => {
    if (!draft.trim() || !sessionId || sending) return;
    const text = draft.trim();
    // Reuse the request id after an ambiguous network failure; never duplicate a task.
    if (request.current?.text !== text)
      request.current = { text, id: crypto.randomUUID() };
    setSending(true);
    setError("");
    try {
      await api(`/sessions/${sessionId}/messages`, {
        text,
        client_id: request.current.id,
      });
      setDraft("");
      request.current = null;
      onSent();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSending(false);
      textarea.current?.focus();
    }
  };
  const stop = async () => {
    try {
      await api(`/sessions/${sessionId}/cancel`, {});
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const suggestions = [
    {
      icon: <FileSearch size={19} />,
      title: "Explore a research project",
      text: "Understand this project, review its findings and research notes, and propose the next experiment with clear validation criteria.",
    },
    {
      icon: <GitBranch size={19} />,
      title: "Build and verify a change",
      text: "Review the current code and open issues, choose a concrete improvement, implement it and validate the result.",
    },
    {
      icon: <Users size={19} />,
      title: "Investigate with a team",
      text: "Investigate the architecture and tests. Delegate independent questions to specialists when useful, then review their evidence and conclusions.",
    },
  ];
  return (
    <div className="workspace-chat">
      <div
        className="conversation-scroll"
        ref={follow.ref}
        onScroll={follow.onScroll}
      >
        {session && session.messages.length > 0 ? (
          <div className="transcript">
            {session.messages.map((message) => (
              <MessageCard key={message.id} message={message} />
            ))}
            {busy && (
              <div className="working-line">
                <span className="pulse-dot" />
                {session.status === "queued"
                  ? "Queued · another session may be running"
                  : session.status === "cancelling"
                    ? "Stopping the current task…"
                    : session.status === "waiting"
                      ? "Waiting for your response"
                      : `${session.phase === "idle" ? "Working" : session.phase.replaceAll("_", " ")}…`}
              </div>
            )}
          </div>
        ) : (
          <div className="welcome">
            <div className="welcome-symbol">
              <Sparkles size={32} strokeWidth={1.3} />
            </div>
            <div className="eyebrow">A workspace for ambitious work</div>
            <h1>
              What are we
              <br />
              <span>working on?</span>
            </h1>
            <p>
              Bring a question, a project, or a difficult problem.
              <br />
              Infinidev will organize the work and bring in specialists when
              needed.
            </p>
            <div className="suggestions">
              {suggestions.map((suggestion) => (
                <button
                  key={suggestion.title}
                  onClick={() => {
                    setDraft(suggestion.text);
                    textarea.current?.focus();
                  }}
                >
                  {suggestion.icon}
                  <span>{suggestion.title}</span>
                  <CornerDownLeft size={14} />
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
      <div className="composer-region">
        {session?.pending.map((pending) => (
          <Question
            key={pending.request_id}
            pending={pending}
            sessionId={sessionId!}
          />
        ))}
        <ErrorNotice error={error} />
        {connection !== "live" && sessionId && (
          <div className="connection-notice">
            {connection === "connecting"
              ? "Connecting to the workspace…"
              : "Connection interrupted. Reconnecting; your task stays on the server."}
          </div>
        )}
        <div className="composer">
          <textarea
            ref={textarea}
            disabled={creating || !sessionId}
            aria-label="Message Infinidev"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder={
              busy
                ? "Add context or steer the work…"
                : "Describe a task. The team starts here."
            }
            onKeyDown={(e) => {
              if (
                e.key === "Enter" &&
                !e.shiftKey &&
                !e.nativeEvent.isComposing
              ) {
                e.preventDefault();
                void send();
              }
            }}
            rows={3}
          />
          <div className="composer-footer">
            <div className="composer-model">
              <Bot size={15} />
              <span>
                {models?.current.split("/").pop() || "Configured model"}
              </span>
              {models?.effort.value && (
                <span className="effort-label">{models.effort.value}</span>
              )}
            </div>
            <div className="composer-actions">
              <button
                className="icon-button"
                aria-label="Scroll to latest message"
                onClick={() => {
                  if (follow.ref.current)
                    follow.ref.current.scrollTop =
                      follow.ref.current.scrollHeight;
                }}
              >
                <ArrowDown size={15} />
              </button>
              {busy && (
                <button
                  className="stop-button"
                  onClick={() => void stop()}
                  aria-label="Stop current task"
                >
                  <Square size={13} fill="currentColor" />
                </button>
              )}
              <button
                className="send-button"
                onClick={() => void send()}
                disabled={
                  !draft.trim() ||
                  sending ||
                  !sessionId ||
                  connection !== "live"
                }
                aria-label={busy ? "Send guidance" : "Send message"}
              >
                {sending ? (
                  <LoaderCircle size={17} className="spin" />
                ) : (
                  <ArrowUp size={20} />
                )}
              </button>
            </div>
          </div>
        </div>
        <div className="composer-hint">
          <span>
            <kbd>↵</kbd> send <span>·</span> <kbd>⇧ ↵</kbd> new line
          </span>
          <span>Runs in your workspace</span>
        </div>
      </div>
    </div>
  );
}

export function Inspector({
  session,
  team,
  onView,
}: {
  session: Session | null;
  team: Team | null;
  onView: (view: View) => void;
}) {
  const agents = Object.entries(team?.board.agents || {});
  const tickets = Object.values(team?.board.tickets || {});
  const notes =
    team?.events
      .filter((e) => e.kind === "note" && !e.superseded_by)
      .slice(-3) || [];
  const complete = tickets.filter((t) => t.status === "accepted").length;
  return (
    <aside className="inspector">
      <div className="inspector-heading">
        <span>Session overview</span>
        <span className="tiny-label">LIVE CONTEXT</span>
      </div>
      <section>
        <div className="section-label">
          Execution
          <Badge status={session?.status || "idle"} />
        </div>
        <p className="inspector-description">
          {session?.phase && session.phase !== "idle"
            ? session.phase.replaceAll("_", " ")
            : "Ready for the next task"}
        </p>
        {tickets.length > 0 && (
          <>
            <div className="progress-track">
              <span
                style={{ width: `${(complete / tickets.length) * 100}%` }}
              />
            </div>
            <small>
              {complete} of {tickets.length} tickets accepted
            </small>
          </>
        )}
      </section>
      <section>
        <button
          className="section-label link-label"
          onClick={() => onView("team")}
        >
          Team{" "}
          <span>
            {agents.length ? `${agents.length} members` : "View team"} →
          </span>
        </button>
        {agents.length ? (
          agents.slice(0, 5).map(([id, agent], i) => (
            <div className="inspector-agent" key={id}>
              <Avatar name={agent.name || id} index={i} small />
              <div>
                <strong>{agent.name || id}</strong>
                <small>{agent.role || "Specialist"}</small>
              </div>
              <span
                className={`member-dot ${team?.live && agent.status === "running" ? "active" : ""}`}
              />
            </div>
          ))
        ) : (
          <div className="inspector-placeholder">
            <Users size={22} />
            <p>
              Specialists will appear here when the orchestrator delegates work.
            </p>
          </div>
        )}
      </section>
      <section>
        <button
          className="section-label link-label"
          onClick={() => onView("team")}
        >
          Shared notes <span>Open →</span>
        </button>
        {notes.length ? (
          notes.map((note) => (
            <div className="mini-note" key={note.id}>
              <small>{note.author_label || note.author}</small>
              <Markdown text={note.content} />
            </div>
          ))
        ) : (
          <div className="inspector-placeholder">
            <MessageSquare size={21} />
            <p>Discoveries and decisions, with the author attached.</p>
          </div>
        )}
      </section>
      <section>
        <div className="section-label">Context</div>
        <div className="context-stat">
          <span>Latest prompt</span>
          <strong>
            {session?.context.prompt_tokens
              ? Number(session.context.prompt_tokens).toLocaleString()
              : "—"}
          </strong>
        </div>
        <small>Actual provider tokens when available.</small>
      </section>
      <div className="inspector-bottom">
        <ShieldCheck size={15} />
        <span>
          Your project. Your tools.
          <br />
          You stay in control.
        </span>
      </div>
    </aside>
  );
}
