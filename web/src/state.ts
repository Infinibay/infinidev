import type { Message, Pending, RecordData, Session } from "./types";
export type StreamEvent = {
  type: string;
  sequence?: number;
  session?: Session;
  message?: Message;
  id?: string;
  chunk?: string;
  pending?: Pending[];
  status?: string;
  phase?: string;
  title?: string;
  steps?: RecordData;
  context?: RecordData;
};
export function reduceSession(
  previous: Session | null,
  event: StreamEvent,
): Session | null {
  if (event.type === "snapshot") return event.session || previous;
  if (
    !previous ||
    (event.sequence !== undefined && event.sequence <= previous.sequence)
  )
    return previous;
  const next = { ...previous, sequence: event.sequence ?? previous.sequence };
  if (event.type === "message" && event.message) {
    const message = event.message;
    const found = next.messages.some((m) => m.id === message.id);
    next.messages = (
      found
        ? next.messages.map((m) => (m.id === message.id ? message : m))
        : [...next.messages, message]
    ).slice(-500);
  } else if (event.type === "delta") {
    next.messages = next.messages.map((m) =>
      m.id === event.id ? { ...m, text: m.text + (event.chunk || "") } : m,
    );
  } else if (event.type === "pending") next.pending = event.pending || [];
  else if (event.type === "state") {
    for (const key of ["status", "phase", "title"] as const)
      if (event[key] !== undefined) next[key] = event[key];
    if (event.steps) next.steps = event.steps;
    if (event.context) next.context = event.context;
  }
  return next;
}
