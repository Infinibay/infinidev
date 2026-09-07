import { describe, expect, it } from "vitest";
import { reduceSession } from "./state";
import type { Session } from "./types";
const original: Session = {
  session_id: "one",
  title: "Research",
  phase: "execute",
  status: "running",
  sequence: 3,
  messages: [{ id: "a", kind: "agent", speaker: "Mara", text: "Evidence " }],
  pending: [],
  steps: {},
  context: {},
};
describe("reconnectable session stream", () => {
  it("ignores pre-snapshot deltas queued during connection", () => {
    expect(
      reduceSession(original, {
        type: "delta",
        sequence: 2,
        id: "a",
        chunk: "duplicate",
      }),
    ).toBe(original);
  });
  it("upserts final messages without duplicating streamed answers", () => {
    const streamed = reduceSession(original, {
      type: "delta",
      sequence: 4,
      id: "a",
      chunk: "first.",
    });
    const final = reduceSession(streamed, {
      type: "message",
      sequence: 5,
      message: {
        id: "a",
        kind: "agent",
        speaker: "Mara",
        text: "Evidence first.",
        streaming: false,
      },
    });
    expect(final?.messages).toHaveLength(1);
    expect(final?.messages[0].text).toBe("Evidence first.");
    expect(original.messages[0].text).toBe("Evidence ");
  });
  it("restores unanswered permissions from a reconnect snapshot", () => {
    const pending = [{ request_id: "p", prompt: "Allow?", kind: "permission" }];
    const current = reduceSession(original, {
      type: "snapshot",
      session: { ...original, pending, status: "waiting", sequence: 10 },
    });
    expect(current?.pending).toEqual(pending);
    const answered = reduceSession(current, {
      type: "pending",
      pending: [],
      sequence: 11,
    });
    expect(answered?.pending).toEqual([]);
  });
});
