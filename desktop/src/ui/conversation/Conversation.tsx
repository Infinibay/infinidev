import { useEffect, useRef } from "react";
import { useEngine } from "@/state/store";
import { Message } from "./Message";
import { Composer } from "./Composer";
import { TaskHeader } from "./TaskHeader";

const EXAMPLES = [
  "Explain what this project does",
  "Find where the agent loop is implemented",
  "Add a CONTRIBUTING.md with build instructions",
];

function EmptyState() {
  const { state, send } = useEngine();
  return (
    <div className="flex h-full flex-col items-center justify-center gap-5 px-6 text-center">
      <div className="text-2xl font-semibold text-fg">Infinidev</div>
      <div className="max-w-md text-sm text-fg-muted">
        An autonomous coding agent running locally on{" "}
        <span className="mono text-fg">{state.model || "your model"}</span>. Ask it to build, fix,
        or explain something — it plans, edits code, runs commands, and shows its work on the right.
      </div>
      <div className="flex flex-wrap justify-center gap-2">
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            onClick={() => send(ex)}
            className="rounded-full border border-fg/12 bg-surface-2/50 px-3 py-1.5 text-xs text-fg-muted hover:border-accent/40 hover:text-fg"
          >
            {ex}
          </button>
        ))}
      </div>
    </div>
  );
}

export function Conversation({ onCommand }: { onCommand?: (action: string) => void }) {
  const { state } = useEngine();
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [state.messages, state.busy]);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <TaskHeader />
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto px-5 py-5">
        {state.messages.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="mx-auto flex max-w-3xl flex-col gap-4">
            {state.messages.map((m) => (
              <Message key={m.id} m={m} />
            ))}
          </div>
        )}
      </div>
      <div className="mx-auto w-full max-w-3xl">
        <Composer onCommand={onCommand} />
      </div>
    </div>
  );
}
