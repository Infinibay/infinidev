import { useState } from "react";
import { Tabs } from "radix-ui";
import {
  ArrowUpRight,
  BookOpen,
  CheckCheck,
  CircleDot,
  MessageSquare,
  Plus,
  Users,
} from "lucide-react";
import { api } from "./api";
import {
  Avatar,
  Badge,
  Empty,
  ErrorNotice,
  JsonDetails,
  Markdown,
  Modal,
  PageTitle,
} from "./components";
import type { Team, TeamAgent, Ticket } from "./types";
import { TeamConversations } from "./TeamConversations";

function IdleDetails({ agent }: { agent: TeamAgent }) {
  if (!agent.waiting) return null;
  const wait = agent.waiting;
  return (
    <div className="idle-details">
      <h3>Idle · waiting for an event</h3>
      <p>{wait.reason}</p>
      <dl>
        <dt>Wake on</dt>
        <dd>{wait.events.map((e) => e.replaceAll("_", " ")).join(", ")}</dd>
        {wait.sender && (
          <>
            <dt>From</dt>
            <dd>{wait.sender}</dd>
          </>
        )}
        {wait.reply_to && (
          <>
            <dt>Answer to</dt>
            <dd>Message #{wait.reply_to}</dd>
          </>
        )}
        {!!wait.task_ids?.length && (
          <>
            <dt>Processes</dt>
            <dd>{wait.task_ids.join(", ")}</dd>
          </>
        )}
        <dt>Timeout</dt>
        <dd>
          {wait.timeout == null
            ? "Until an event arrives"
            : `${wait.timeout} seconds`}
        </dd>
      </dl>
      <small>
        No model calls while idle. User guidance and cancellation interrupt the
        wait.
      </small>
    </div>
  );
}

function AssignmentDetails({ value }: { value: TeamAgent | Ticket }) {
  if (typeof value.role === "string")
    return (
      <div className="assignment-details">
        <div className="assignment-role">
          <Badge status={String(value.status || "idle")} />
          <span>{value.role}</span>
        </div>
        <IdleDetails agent={value as TeamAgent} />
        <h3>Granted tools</h3>
        <div className="tool-grants">
          {(Array.isArray(value.tools) ? value.tools : []).map((tool) => (
            <code key={String(tool)}>{String(tool)}</code>
          ))}
        </div>
        {typeof value.system_prompt === "string" && (
          <>
            <h3>Specialist instructions</h3>
            <Markdown text={value.system_prompt} />
          </>
        )}
        {typeof value.last_result === "string" && (
          <>
            <h3>Latest report</h3>
            <Markdown text={value.last_result} />
          </>
        )}
        <JsonDetails label="Runtime details" value={value} />
      </div>
    );
  return (
    <div className="assignment-details">
      <Badge status={String(value.status || "pending")} />
      <h3>Objective</h3>
      <Markdown
        text={String(
          value.objective || value.description || "No description recorded.",
        )}
      />
      {Array.isArray(value.acceptance) && (
        <>
          <h3>Acceptance criteria</h3>
          <ul>
            {value.acceptance.map((criterion, index) => (
              <li key={index}>{String(criterion)}</li>
            ))}
          </ul>
        </>
      )}
      {Array.isArray(value.constraints) && value.constraints.length > 0 && (
        <>
          <h3>Constraints</h3>
          <ul>
            {value.constraints.map((constraint, index) => (
              <li key={index}>{String(constraint)}</li>
            ))}
          </ul>
        </>
      )}
      {Boolean(value.result) && (
        <>
          <h3>Report</h3>
          <Markdown text={String(value.result)} />
        </>
      )}
      {Boolean(value.review) && (
        <>
          <h3>Review</h3>
          <Markdown text={String(value.review)} />
        </>
      )}
      <JsonDetails label="Ticket details and dependencies" value={value} />
    </div>
  );
}

export function TeamView({
  team,
  sessionId,
  error,
  ask,
}: {
  team: Team | null;
  sessionId: string | null;
  error: string;
  ask: (text: string) => void;
}) {
  const [selected, setSelected] = useState<{
    title: string;
    value: TeamAgent | Ticket;
  } | null>(null);
  const [noteOpen, setNoteOpen] = useState(false);
  const [note, setNote] = useState("");
  const [noteError, setNoteError] = useState("");
  const [saving, setSaving] = useState(false);
  const agents = Object.entries(team?.board.agents || {});
  const tickets = Object.entries(team?.board.tickets || {});
  const selectedValue = selected?.value.id
    ? team?.board.agents?.[selected.value.id] ||
      team?.board.tickets?.[selected.value.id] ||
      selected.value
    : selected?.value;
  const groups = [
    {
      name: "To do",
      statuses: ["pending", "interrupted", "blocked", "needs_work"],
    },
    { name: "In progress", statuses: ["running"] },
    { name: "In review", statuses: ["review"] },
    { name: "Resolved", statuses: ["accepted", "cancelled", "failed"] },
  ];
  async function saveNote() {
    setSaving(true);
    setNoteError("");
    try {
      await api(`/sessions/${sessionId}/notes`, { content: note });
      setNote("");
      setNoteOpen(false);
    } catch (e) {
      setNoteError((e as Error).message);
    } finally {
      setSaving(false);
    }
  }
  return (
    <div className="page-scroll">
      <PageTitle
        eyebrow="Collective intelligence"
        title="A team with a shared purpose."
        actions={
          <button
            className="button"
            onClick={() =>
              ask(
                "Review the team’s current progress, unresolved questions, and next steps.",
              )
            }
          >
            <MessageSquare size={15} />
            Ask the orchestrator
          </button>
        }
      >
        Follow the people doing the work, the evidence they collect, and the
        decisions they make.
      </PageTitle>
      <ErrorNotice error={error} />
      {!agents.length ? (
        <Empty
          icon={<Users size={28} />}
          title="The right specialists, when needed"
          action={
            <button
              className="button primary"
              onClick={() =>
                ask(
                  "Investigate this project and delegate independent research questions to specialists when useful.",
                )
              }
            >
              <Plus size={15} />
              Start a task
            </button>
          }
        >
          Give Infinidev a task in the conversation. The orchestrator chooses
          names, roles, prompts and tools for its specialists.
        </Empty>
      ) : (
        <>
          <div className="team-summary">
            <span className="eyebrow">{agents.length} team members</span>
            <Badge status={team?.live ? "running" : "idle"}>
              {team?.live ? "Team active" : "Saved team · no active run"}
            </Badge>
          </div>
          <div className="agent-grid">
            {agents.map(([id, agent], index) => (
              <button
                className="agent-card"
                key={id}
                onClick={() =>
                  setSelected({ title: agent.name || id, value: agent })
                }
              >
                <div className="agent-card-top">
                  <Avatar name={agent.name || id} index={index} />
                  <Badge status={team?.live ? agent.status : "idle"} />
                </div>
                <h3>{agent.name || id}</h3>
                <p>{agent.role || "Specialist"}</p>
                {team?.live && agent.waiting && (
                  <div className="idle-hint">Idle · {agent.waiting.reason}</div>
                )}
                <div className="agent-card-footer">
                  <span>
                    {id === "orchestrator"
                      ? "Coordinates the team"
                      : `${agent.tools?.length || 0} tools available`}
                  </span>
                  <ArrowUpRight size={15} />
                </div>
              </button>
            ))}
          </div>
          <Tabs.Root defaultValue="tickets" className="team-tabs">
            <Tabs.List className="tabs-list" aria-label="Team information">
              <Tabs.Trigger value="tickets">
                <CheckCheck size={16} />
                Tickets <span>{tickets.length}</span>
              </Tabs.Trigger>
              <Tabs.Trigger value="notes">
                <BookOpen size={16} />
                Shared notes
              </Tabs.Trigger>
              <Tabs.Trigger value="messages">
                <MessageSquare size={16} />
                Communication
              </Tabs.Trigger>
              <Tabs.Trigger value="activity">
                <CircleDot size={16} /> Activity
              </Tabs.Trigger>
            </Tabs.List>
            <Tabs.Content value="tickets">
              <div className="kanban">
                {groups.map((group) => {
                  const rows = tickets.filter(([, t]) =>
                    group.statuses.includes(t.status || "pending"),
                  );
                  return (
                    <section className="kanban-column" key={group.name}>
                      <div className="kanban-title">
                        <CircleDot size={13} />
                        <strong>{group.name}</strong>
                        <span>{rows.length}</span>
                      </div>
                      {rows.map(([id, ticket]) => (
                        <button
                          className="ticket-card"
                          key={id}
                          onClick={() =>
                            setSelected({
                              title: ticket.title || id,
                              value: ticket,
                            })
                          }
                        >
                          <small>{id.slice(0, 10)}</small>
                          <h4>{ticket.title || "Untitled ticket"}</h4>
                          <p>
                            {String(
                              ticket.description || ticket.objective || "",
                            )}
                          </p>
                          <div>
                            <span>
                              {team?.board.agents?.[ticket.assignee || ""]
                                ?.name || "Unassigned"}
                            </span>
                            <Badge status={ticket.status} />
                          </div>
                        </button>
                      ))}
                      {!rows.length && (
                        <p className="column-empty">No tickets here</p>
                      )}
                    </section>
                  );
                })}
              </div>
            </Tabs.Content>
            <Tabs.Content value="notes">
              <div className="tab-toolbar">
                <span>Shared discoveries, always attributed.</span>
                <button className="button" onClick={() => setNoteOpen(true)}>
                  <Plus size={15} />
                  Add a note
                </button>
              </div>
              <div className="notes-grid">
                {team?.events
                  .filter((e) => e.kind === "note" && !e.superseded_by)
                  .map((event) => (
                    <article className="note-card" key={event.id}>
                      <div>
                        <Avatar
                          name={event.author_label || event.author}
                          small
                        />
                        <strong>{event.author_label || event.author}</strong>
                        <time>
                          {new Date(event.created_at).toLocaleDateString()}
                        </time>
                      </div>
                      <Markdown text={event.content} />
                    </article>
                  ))}
              </div>
              {!team?.events.some((e) => e.kind === "note") && (
                <Empty
                  icon={<BookOpen size={25} />}
                  title="A shared memory for the team"
                >
                  Findings, caveats and decisions will collect here. Add context
                  of your own with your authorship preserved.
                </Empty>
              )}
            </Tabs.Content>
            <Tabs.Content value="messages">
              {team && <TeamConversations team={team} sessionId={sessionId} />}
            </Tabs.Content>
            <Tabs.Content value="activity">
              <div className="team-timeline">
                {team?.events
                  .filter((e) => e.kind !== "note" && e.kind !== "message")
                  .slice()
                  .reverse()
                  .map((event) => (
                    <article key={event.id}>
                      <Avatar name={event.author_label || event.author} small />
                      <div>
                        <div className="timeline-meta">
                          <strong>{event.author_label || event.author}</strong>
                          {event.recipient_label && (
                            <span>→ {event.recipient_label}</span>
                          )}
                          <span className="tiny-label">{event.kind}</span>
                          <time>
                            {new Date(event.created_at).toLocaleTimeString([], {
                              hour: "2-digit",
                              minute: "2-digit",
                            })}
                          </time>
                        </div>
                        <Markdown text={event.content} />
                        {event.ticket_id && (
                          <small>Ticket {event.ticket_id}</small>
                        )}
                      </div>
                    </article>
                  ))}
              </div>
              {!team?.events.length && (
                <Empty
                  icon={<MessageSquare size={25} />}
                  title="The conversation behind the work"
                >
                  Messages and requests between team members appear here as they
                  happen.
                </Empty>
              )}
            </Tabs.Content>
          </Tabs.Root>
        </>
      )}
      <Modal
        open={!!selected}
        onOpenChange={(open) => {
          if (!open) setSelected(null);
        }}
        title={selected?.title || "Details"}
        description="Assignment, scope and current results."
      >
        {selected && (
          <>
            <AssignmentDetails value={selectedValue || selected.value} />
            <button
              className="button primary"
              onClick={() => {
                ask(
                  `Ask ${selected.title} about their current progress and any unresolved questions.`,
                );
                setSelected(null);
              }}
            >
              Discuss in the conversation <ArrowUpRight size={15} />
            </button>
          </>
        )}
      </Modal>
      <Modal
        open={noteOpen}
        onOpenChange={setNoteOpen}
        title="Add a shared note"
        description="This note is saved as You and is readable by the team."
      >
        <textarea
          className="note-editor"
          aria-label="Shared note"
          placeholder="A useful finding, decision, or constraint…"
          value={note}
          onChange={(e) => setNote(e.target.value)}
        />
        <ErrorNotice error={noteError} />
        <div className="question-actions">
          <button
            className="button primary"
            disabled={saving || !note.trim()}
            onClick={() => void saveNote()}
          >
            {saving ? "Saving…" : "Save note"}
          </button>
        </div>
      </Modal>
    </div>
  );
}
