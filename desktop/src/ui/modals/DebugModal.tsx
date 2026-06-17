import { useEffect, useState } from "react";
import { Dialog } from "@harbor/components/overlays/Dialog";
import { Badge } from "@harbor/components/display/Badge";
import { useAppState, useEngine } from "@/state/store";
import type { ChatMessage } from "@/state/store";
import type { NoteRow } from "@/api/engineEvent";

interface Props {
  open: boolean;
  onClose: () => void;
}

const SECTIONS = ["Notes", "History", "Plan", "State"] as const;
type Section = (typeof SECTIONS)[number];

/** A one-line argument summary for a tool call (path / command / pattern). */
function argSummary(args: unknown): string {
  const a = args as Record<string, unknown> | null;
  const v = a && (a.path ?? a.command ?? a.pattern ?? a.query ?? a.note);
  return typeof v === "string" ? v : "";
}

/** Inspect agent internals — a port of the TUI debug panel. Reads the live
 *  store state (decisions, tool calls, plan, usage) plus the persisted notes,
 *  giving a single place to see what the agent did and is holding in memory. */
export function DebugModal({ open, onClose }: Props) {
  const s = useAppState();
  const { commands } = useEngine();
  const [section, setSection] = useState<Section>("Notes");
  const [notes, setNotes] = useState<NoteRow[]>([]);

  useEffect(() => {
    if (!open) return;
    commands
      .notesList()
      .then(setNotes)
      .catch(() => setNotes([]));
  }, [open, commands, s.projectNonce]);

  const tools = s.messages.filter((m) => m.kind === "tool");

  return (
    <Dialog open={open} onClose={onClose} size="lg" title="Debug — agent internals">
      <div className="grid h-[60vh] grid-cols-[140px_1fr] gap-3">
        <div className="space-y-0.5 border-r border-fg/8 pr-2">
          {SECTIONS.map((sec) => (
            <button
              key={sec}
              onClick={() => setSection(sec)}
              className={`flex w-full items-center justify-between rounded px-2 py-1.5 text-left text-[12px] ${
                section === sec ? "bg-accent/15 text-fg" : "text-fg-muted hover:bg-surface-2/60"
              }`}
            >
              <span>{sec}</span>
              <span className="text-[10px] text-fg-subtle">{counts(sec, notes.length, tools.length, s.plan.length)}</span>
            </button>
          ))}
        </div>

        <div className="overflow-y-auto pr-1 text-[12px]">
          {section === "Notes" && <NotesSection notes={notes} />}
          {section === "History" && <HistorySection tools={tools} decision={s.decision} review={s.review} />}
          {section === "Plan" && <PlanSection plan={s.plan} />}
          {section === "State" && <StateSection s={s} />}
        </div>
      </div>
    </Dialog>
  );
}

function counts(sec: Section, notes: number, tools: number, plan: number): string {
  if (sec === "Notes") return notes ? String(notes) : "";
  if (sec === "History") return tools ? String(tools) : "";
  if (sec === "Plan") return plan ? String(plan) : "";
  return "";
}

function NotesSection({ notes }: { notes: NoteRow[] }) {
  if (!notes.length) return <Empty>No notes recorded.</Empty>;
  return (
    <ol className="space-y-1.5">
      {notes.map((n, i) => (
        <li key={n.id} className="flex gap-2">
          <span className="mono text-fg-subtle">{i + 1}.</span>
          <span className="text-fg">{n.note}</span>
        </li>
      ))}
    </ol>
  );
}

function HistorySection({
  tools,
  decision,
  review,
}: {
  tools: ChatMessage[];
  decision: { kind: string; detail: string } | null;
  review: { status: string; notes: string[] } | null;
}) {
  if (!tools.length && !decision && !review) return <Empty>No steps completed yet.</Empty>;
  return (
    <div className="space-y-1.5">
      {decision && (
        <div className="rounded border border-fg/8 bg-surface-2/40 px-2 py-1.5">
          <Badge tone={decision.kind === "escalate" ? "warning" : "neutral"}>{decision.kind}</Badge>
          {decision.detail && <span className="ml-2 text-fg-muted">{decision.detail}</span>}
        </div>
      )}
      {tools.map((t) => {
        const arg = argSummary(t.toolArgs);
        return (
          <div key={t.id} className="rounded border border-fg/8 bg-surface-2/40 px-2 py-1.5">
            <div className="flex items-center gap-2">
              <span className={t.toolDone ? (t.toolOk ? "text-success" : "text-danger") : "text-fg-subtle"}>
                {t.toolDone ? (t.toolOk ? "✓" : "✕") : "…"}
              </span>
              <span className="mono text-fg">{t.toolName}</span>
              {arg && <span className="mono truncate text-[11px] text-fg-subtle">{arg}</span>}
            </div>
            {t.toolOutput && (
              <pre className="mono mt-1 max-h-24 overflow-y-auto whitespace-pre-wrap text-[10px] text-fg-subtle">
                {t.toolOutput.slice(0, 600)}
              </pre>
            )}
          </div>
        );
      })}
      {review && (
        <div className="rounded border border-fg/8 bg-surface-2/40 px-2 py-1.5">
          <Badge tone={review.status === "approve" ? "success" : "warning"}>review: {review.status}</Badge>
          {review.notes.length > 0 && (
            <ul className="ml-4 mt-1 list-disc text-fg-muted">
              {review.notes.map((n, i) => (
                <li key={i}>{n}</li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

function PlanSection({ plan }: { plan: string[] }) {
  if (!plan.length) return <Empty>No plan for this turn.</Empty>;
  return (
    <ol className="list-decimal space-y-1 pl-5">
      {plan.map((p, i) => (
        <li key={i} className="text-fg">
          {p}
        </li>
      ))}
    </ol>
  );
}

function StateSection({ s }: { s: ReturnType<typeof useAppState> }) {
  const rows: [string, string][] = [
    ["Model", s.model || "—"],
    ["Phase", s.phase],
    ["Busy", s.busy ? "yes" : "no"],
    ["Step", s.step ? `${s.step.n} / ${s.step.max}` : "—"],
    ["Prompt tokens", s.usage ? String(s.usage.prompt) : "—"],
    ["Completion tokens", s.usage ? String(s.usage.completion) : "—"],
    ["Total tokens", s.usage ? String(s.usage.total) : "—"],
    ["Cost (USD)", s.costUsd ? `$${s.costUsd.toFixed(4)}` : "—"],
    ["Changed files", String(s.changedFiles.length)],
    ["Messages", String(s.messages.length)],
  ];
  return (
    <table className="w-full text-[12px]">
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k} className="border-b border-fg/6">
            <td className="py-1 pr-3 text-fg-subtle">{k}</td>
            <td className="mono py-1 text-fg">{v}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return <div className="grid h-full place-items-center text-[12px] text-fg-subtle">{children}</div>;
}
