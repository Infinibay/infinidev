import { useState } from "react";
import { MarkdownRenderer } from "@harbor/components/dev/MarkdownRenderer";
import { CodeBlock } from "@harbor/components/dev/CodeBlock";
import { Badge } from "@harbor/components/display/Badge";
import { Spinner } from "@harbor/components/display/Spinner";
import type { ChatMessage } from "@/state/store";

function asText(v: unknown): string {
  if (v == null) return "";
  if (typeof v === "string") return v;
  try {
    return JSON.stringify(v, null, 2);
  } catch {
    return String(v);
  }
}

function argSummary(args: unknown): string {
  if (args && typeof args === "object") {
    const a = args as Record<string, unknown>;
    const key = a.path ?? a.command ?? a.pattern ?? a.query;
    if (typeof key === "string") return key;
  }
  return "";
}

function ToolCard({ m }: { m: ChatMessage }) {
  const [open, setOpen] = useState(false);
  const running = !m.toolDone;
  const failed = m.toolDone && m.toolOk === false;
  const tone = failed ? "danger" : running ? "info" : "success";
  // A tone-coloured left spine gives each card status-at-a-glance and keeps the
  // activity stream from reading as a flat wall of grey rounded rects.
  const edge = failed ? "border-l-danger/70" : running ? "border-l-info/70" : "border-l-success/55";
  return (
    <div className={`rounded-lg border border-fg/10 border-l-2 ${edge} bg-surface-2/50`}>
      <button className="flex w-full items-center gap-2 px-3 py-2 text-left" onClick={() => setOpen((o) => !o)}>
        <Badge tone={tone}>{m.toolName ?? "tool"}</Badge>
        <span className="mono truncate text-xs text-fg-muted">{argSummary(m.toolArgs)}</span>
        <span className="ml-auto flex items-center gap-2 text-fg-subtle">
          {running ? <Spinner /> : <span className="text-[10px]">{open ? "▾" : "▸"}</span>}
        </span>
      </button>
      {open && (
        <div className="space-y-2 border-t border-fg/10 p-3">
          {argSummary(m.toolArgs) === "" && (
            <CodeBlock code={asText(m.toolArgs)} lang="json" showLineNumbers={false} />
          )}
          {m.toolOutput && (
            <CodeBlock
              code={m.toolOutput}
              lang="text"
              showLineNumbers={false}
              className={failed ? "border border-danger/40" : undefined}
            />
          )}
        </div>
      )}
    </div>
  );
}

export function Message({ m }: { m: ChatMessage }) {
  switch (m.kind) {
    case "user":
      return (
        <div className="flex flex-col items-end gap-1.5">
          {m.images && m.images.length > 0 && (
            <div className="flex max-w-[82%] flex-wrap justify-end gap-1.5">
              {m.images.map((src, i) => (
                <img
                  key={i}
                  src={src}
                  alt={`attachment ${i + 1}`}
                  className="h-24 w-24 rounded-lg border border-fg/15 object-cover"
                />
              ))}
            </div>
          )}
          {m.text && (
            <div className="max-w-[82%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-accent/90 px-4 py-2.5 text-sm text-white shadow-[0_4px_18px_-6px_rgb(var(--harbor-accent)/0.55)]">
              {m.text}
            </div>
          )}
        </div>
      );
    case "system":
      return <div className="px-1 text-xs italic text-fg-subtle">{m.text}</div>;
    case "error":
      return (
        <div className="rounded-lg border border-danger/40 bg-danger/10 px-3.5 py-2.5 text-sm text-danger">
          {m.text}
        </div>
      );
    case "tool":
      return <ToolCard m={m} />;
    case "agent":
    default:
      return (
        <div className="max-w-[92%] text-sm leading-relaxed text-fg">
          {m.streaming ? (
            <span className="whitespace-pre-wrap">
              {m.text}
              <span className="ml-0.5 inline-block h-3.5 w-[3px] animate-pulse bg-accent align-middle" />
            </span>
          ) : (
            <MarkdownRenderer source={m.text} />
          )}
        </div>
      );
  }
}
