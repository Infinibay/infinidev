import { useAppState } from "@/state/store";
import { contextWindow } from "../contextWindow";

function Heading({ children }: { children: React.ReactNode }) {
  return (
    <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-fg-subtle">{children}</div>
  );
}

/** Context-window usage: how much of the model's window the last prompt used. */
function ContextMeter({ used, model }: { used: number; model: string }) {
  const window = contextWindow(model);
  const pct = Math.min(100, Math.round((used / window) * 100));
  const tone = pct >= 90 ? "bg-danger" : pct >= 70 ? "bg-warning" : "bg-accent";
  return (
    <section>
      <Heading>Context</Heading>
      <div className="mb-1 flex items-baseline justify-between text-[11px]">
        <span className="text-fg-muted">
          {used.toLocaleString()} / {window.toLocaleString()} tok
        </span>
        <span className="text-fg-subtle">{pct}%</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-2/60">
        <div className={`h-full rounded-full ${tone} transition-all`} style={{ width: `${pct}%` }} />
      </div>
    </section>
  );
}

export function ActivityTab() {
  const s = useAppState();
  const tools = s.messages.filter((m) => m.kind === "tool");

  return (
    <div className="h-full space-y-5 overflow-y-auto p-4 text-sm">
      {s.decision && (
        <section>
          <Heading>Triage</Heading>
          {s.decision.kind === "respond" ? (
            <div className="flex items-center gap-2 text-xs text-fg-muted">
              <span>💬</span>
              <span>Answered directly — no developer run needed.</span>
            </div>
          ) : (
            <div className="flex items-start gap-2 text-xs text-fg-muted">
              <span>🚀</span>
              <span>
                <span className="text-accent">Escalated to the developer.</span>
                {s.decision.detail && <span className="block text-fg-subtle">{s.decision.detail}</span>}
              </span>
            </div>
          )}
        </section>
      )}

      {s.plan.length > 0 && (
        <section>
          <Heading>Plan</Heading>
          <ol className="space-y-1">
            {s.plan.map((step, i) => (
              <li key={i} className="flex gap-2 text-xs text-fg-muted">
                <span className="shrink-0 text-fg-subtle">{i + 1}.</span>
                <span>{step}</span>
              </li>
            ))}
          </ol>
        </section>
      )}

      <section>
        <Heading>Status</Heading>
        {s.busy ? (
          <div className="text-accent">
            {s.phase}
            {s.step && <span className="text-fg-subtle"> · step {s.step.n}/{s.step.max}</span>}
          </div>
        ) : (
          <div className="text-fg-subtle">Idle</div>
        )}
      </section>

      {s.usage && s.usage.prompt > 0 && <ContextMeter used={s.usage.prompt} model={s.model} />}

      <section>
        <Heading>Tool activity</Heading>
        {tools.length === 0 ? (
          <div className="text-xs text-fg-subtle">No tools run yet this session.</div>
        ) : (
          <ul className="space-y-1.5">
            {tools.slice(-40).map((t) => {
              const running = !t.toolDone;
              const failed = t.toolDone && t.toolOk === false;
              const arg = ((): string => {
                const a = t.toolArgs as Record<string, unknown> | null;
                const v = a && (a.path ?? a.command ?? a.pattern ?? a.query);
                return typeof v === "string" ? v : "";
              })();
              return (
                <li key={t.id} className="flex items-center gap-2 text-xs">
                  <span className={running ? "text-info" : failed ? "text-danger" : "text-success"}>
                    {running ? "◌" : failed ? "✗" : "✓"}
                  </span>
                  <span className="text-fg-muted">{t.toolName}</span>
                  <span className="mono truncate text-fg-subtle">{arg}</span>
                </li>
              );
            })}
          </ul>
        )}
      </section>

      {s.review && (
        <section>
          <Heading>Review</Heading>
          {s.review.status === "approve" ? (
            <div className="flex items-center gap-2 text-xs text-success">
              <span>✓</span>
              <span>Reviewer approved the changes.</span>
            </div>
          ) : (
            <div className="space-y-1.5">
              <div className="flex items-center gap-2 text-xs text-warning">
                <span>⚠</span>
                <span>Reviewer requested changes — applying a fix pass.</span>
              </div>
              <ul className="space-y-1 pl-1">
                {s.review.notes.map((n, i) => (
                  <li key={i} className="flex gap-2 text-xs text-fg-muted">
                    <span className="shrink-0 text-fg-subtle">·</span>
                    <span>{n}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}

      {s.reasoning.trim() && (
        <section>
          <Heading>Reasoning</Heading>
          <pre className="mono max-h-72 overflow-auto whitespace-pre-wrap rounded-md bg-surface-2/40 p-3 text-[11px] text-fg-subtle">
            {s.reasoning}
          </pre>
        </section>
      )}
    </div>
  );
}
