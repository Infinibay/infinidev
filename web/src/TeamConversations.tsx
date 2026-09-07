import { useState } from "react";
import { MessageSquare, Search } from "lucide-react";
import {
  Avatar,
  Badge,
  Empty,
  ErrorNotice,
  Markdown,
  Modal,
} from "./components";
import { useResource } from "./hooks";
import type { Team, TeamEvent } from "./types";

function delivery(message: TeamEvent, team: Team) {
  if (
    message.delivery === "answered" ||
    team.events.some((e) => e.reply_to === message.id)
  )
    return "answered";
  const targets = Object.entries(team.board.agents || {}).filter(
    ([id]) =>
      id !== message.author &&
      (message.recipient === "all" || id === message.recipient),
  );
  return targets.length &&
    targets.every(
      ([, member]) =>
        (member.cursor || 0) >= message.id ||
        member.received?.includes(message.id),
    )
    ? "delivered"
    : message.delivery || "queued";
}

function Message({ event, team }: { event: TeamEvent; team: Team }) {
  return (
    <article className="peer-message">
      <Avatar name={event.author_label || event.author} small />
      <div>
        <div className="timeline-meta">
          <strong>{event.author_label || event.author}</strong>
          <span>→ {event.recipient_label || event.recipient}</span>
          <span className="tiny-label">
            {event.message_type || (event.reply_to ? "reply" : "request")}
          </span>
          <Badge status={delivery(event, team)} />
          <time>
            {new Date(event.created_at).toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            })}
          </time>
        </div>
        {event.reply_to && <small>Reply to #{event.reply_to}</small>}
        <Markdown text={event.content} />
      </div>
    </article>
  );
}

function Thread({
  sessionId,
  threadId,
  team,
}: {
  sessionId: string;
  threadId: number;
  team: Team;
}) {
  const [after, setAfter] = useState(0);
  const { data, error } = useResource<{
    events: TeamEvent[];
    next: number;
    has_more: boolean;
  }>(`/sessions/${sessionId}/threads/${threadId}?after=${after}`, 2000);
  return (
    <>
      <ErrorNotice error={error} />
      {!data && !error && <p>Loading conversation…</p>}
      {data?.events.map((event) => (
        <Message key={event.id} event={event} team={team} />
      ))}
      <div className="question-actions">
        {after > 0 && (
          <button className="button" onClick={() => setAfter(0)}>
            Back to the question
          </button>
        )}
        {data?.has_more && (
          <button className="button" onClick={() => setAfter(data.next)}>
            Next messages
          </button>
        )}
      </div>
    </>
  );
}

export function TeamConversations({
  team,
  sessionId,
}: {
  team: Team;
  sessionId: string | null;
}) {
  const [query, setQuery] = useState("");
  const [participant, setParticipant] = useState("");
  const [selected, setSelected] = useState<number | null>(null);
  const threads = new Map<number, TeamEvent[]>();
  for (const event of team.events.filter((e) => e.kind === "message")) {
    const id = event.thread_id || event.reply_to || event.id;
    threads.set(id, [...(threads.get(id) || []), event]);
  }
  const visible = [...threads]
    .filter(
      ([, events]) =>
        (!participant ||
          events.some(
            (e) =>
              e.author === participant ||
              e.recipient === participant ||
              e.recipient === "all",
          )) &&
        (!query ||
          events.some((e) =>
            `${e.content} ${e.author_label} ${e.recipient_label}`
              .toLowerCase()
              .includes(query.toLowerCase()),
          )),
    )
    .reverse();
  return (
    <>
      <div className="tab-toolbar conversation-filters">
        <label>
          <Search size={15} />
          <input
            aria-label="Search team conversations"
            placeholder="Find a question or answer…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </label>
        <select
          aria-label="Conversation participant"
          value={participant}
          onChange={(e) => setParticipant(e.target.value)}
        >
          <option value="">Everyone</option>
          {Object.entries(team.board.agents || {}).map(([id, agent]) => (
            <option key={id} value={id}>
              {agent.name || id}
            </option>
          ))}
        </select>
      </div>
      <div className="conversation-threads">
        {visible.map(([id, events]) => (
          <section
            className="conversation-thread"
            key={id}
            aria-label={`Conversation ${id}`}
          >
            <div className="thread-heading">
              <span>
                <MessageSquare size={15} /> Conversation #{id}
                {events[0].ticket_id &&
                  ` · ${team.board.tickets?.[events[0].ticket_id]?.title || events[0].ticket_id}`}
              </span>
              <button className="button quiet" onClick={() => setSelected(id)}>
                Open full thread
              </button>
            </div>
            {events[0].id !== id && (
              <p className="muted">
                Open the full thread to read earlier context.
              </p>
            )}
            {events.map((event) => (
              <Message key={event.id} event={event} team={team} />
            ))}
          </section>
        ))}
      </div>
      {!visible.length && (
        <Empty
          icon={<MessageSquare size={25} />}
          title="No matching conversations"
        >
          Questions and their answers stay together here, with authorship and
          delivery status.
        </Empty>
      )}
      <Modal
        open={selected !== null}
        onOpenChange={(open) => {
          if (!open) setSelected(null);
        }}
        title={`Conversation #${selected}`}
        description="The original question, its replies and their delivery status."
      >
        {selected !== null && sessionId && (
          <Thread
            key={selected}
            sessionId={sessionId}
            threadId={selected}
            team={team}
          />
        )}
      </Modal>
    </>
  );
}
