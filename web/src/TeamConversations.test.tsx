import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { TeamConversations } from "./TeamConversations";
import { TeamView } from "./TeamView";
import type { Team } from "./types";

const team: Team = {
  live: true,
  next: 3,
  has_more: false,
  board: {
    agents: {
      a: {
        id: "a",
        name: "Alice",
        role: "Developer",
        status: "waiting",
        cursor: 3,
        waiting: {
          events: ["message"],
          reply_to: 1,
          reason: "Bob is checking gradients",
          since: "2026-09-07T01:00:00Z",
        },
      },
      b: {
        id: "b",
        name: "Bob",
        role: "Researcher",
        status: "running",
        cursor: 3,
      },
    },
    tickets: {},
  },
  events: [
    {
      id: 1,
      kind: "message",
      author: "a",
      author_label: "Alice",
      recipient: "b",
      recipient_label: "Bob",
      content: "Does the cache detach?",
      thread_id: 1,
      message_type: "request",
      created_at: "2026-09-07T01:00:00Z",
    },
    {
      id: 2,
      kind: "message",
      author: "b",
      author_label: "Bob",
      recipient: "a",
      recipient_label: "Alice",
      content: "Yes, cache.py:42 detaches it.",
      thread_id: 1,
      reply_to: 1,
      message_type: "reply",
      created_at: "2026-09-07T01:00:01Z",
    },
    {
      id: 3,
      kind: "message",
      author: "b",
      recipient: "all",
      content: "Evaluation started.",
      thread_id: 3,
      message_type: "info",
      created_at: "2026-09-07T01:00:02Z",
    },
  ],
};

describe("team coordination", () => {
  it("keeps answers with their question and filters complete threads", async () => {
    render(<TeamConversations team={team} sessionId="session" />);
    const conversation = screen.getByRole("region", { name: "Conversation 1" });
    expect(
      within(conversation).getByText("Does the cache detach?"),
    ).toBeVisible();
    expect(
      within(conversation).getByText("Yes, cache.py:42 detaches it."),
    ).toBeVisible();
    expect(within(conversation).getByText("answered")).toBeVisible();
    await userEvent.type(
      screen.getByRole("textbox", { name: "Search team conversations" }),
      "cache.py",
    );
    expect(screen.getByText("Does the cache detach?")).toBeVisible();
    expect(screen.queryByText("Evaluation started.")).not.toBeInTheDocument();
  });

  it("shows the idle reason and the agent's selected wake events", async () => {
    render(
      <TeamView team={team} sessionId="session" error="" ask={() => {}} />,
    );
    expect(screen.getByText("Idle · Bob is checking gradients")).toBeVisible();
    await userEvent.click(screen.getByRole("heading", { name: /^Alice$/ }));
    const dialog = screen.getByRole("dialog");
    expect(
      within(dialog).getByText("Idle · waiting for an event"),
    ).toBeVisible();
    expect(within(dialog).getByText("Message #1")).toBeVisible();
    expect(within(dialog).getByText("Until an event arrives")).toBeVisible();
  });
});
