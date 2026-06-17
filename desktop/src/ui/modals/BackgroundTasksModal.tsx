import { useEffect, useState } from "react";
import { Dialog } from "@harbor/components/overlays/Dialog";
import { Badge } from "@harbor/components/display/Badge";
import { Button } from "@harbor/components/buttons/Button";
import { useEngine } from "@/state/store";
import type { BgTask } from "@/api/engineEvent";

interface Props {
  open: boolean;
  onClose: () => void;
}

/** Browser for background tasks the agent started with `run_in_background`
 *  (dev servers, watchers, builds). Mirrors the TUI's background-tasks
 *  explorer: polls the live registry so a server that just printed its
 *  readiness line or a watcher that just exited shows up within ~1s. */
export function BackgroundTasksModal({ open, onClose }: Props) {
  const { commands } = useEngine();
  const [tasks, setTasks] = useState<BgTask[]>([]);

  useEffect(() => {
    if (!open) return;
    let alive = true;
    const tick = () =>
      commands
        .bgList()
        .then((t) => alive && setTasks(t))
        .catch(() => alive && setTasks([]));
    tick();
    const h = setInterval(tick, 1200);
    return () => {
      alive = false;
      clearInterval(h);
    };
  }, [open, commands]);

  const running = tasks.filter((t) => t.running).length;

  const stop = (id: number) => {
    void commands.bgStop(id).then(() => commands.bgList().then(setTasks));
  };

  return (
    <Dialog open={open} onClose={onClose} size="lg" title="Background Tasks">
      <div className="text-[10px] text-fg-subtle">
        {tasks.length} total · {running} running · refreshes live
      </div>
      <div className="mt-2 max-h-[60vh] space-y-2 overflow-y-auto pr-1">
        {tasks.length === 0 && (
          <div className="px-2 py-8 text-center text-[12px] text-fg-subtle">
            No background tasks. The agent starts them with
            <span className="mono"> run_in_background</span> — dev servers, watchers, builds.
          </div>
        )}
        {tasks.map((t) => (
          <div key={t.id} className="rounded-md border border-fg/10 bg-surface-2/40 p-2.5">
            <div className="flex items-center gap-2">
              <Badge tone={t.running ? "success" : t.exit_code === 0 ? "neutral" : "danger"}>
                {t.status}
              </Badge>
              <span className="mono truncate text-[12px] text-fg">
                #{t.id} {t.description}
              </span>
              <div className="ml-auto">
                {t.running && (
                  <Button variant="destructive" size="sm" onClick={() => stop(t.id)}>
                    Stop
                  </Button>
                )}
              </div>
            </div>
            <div className="mono mt-1 truncate text-[10px] text-fg-subtle">$ {t.command}</div>
            {t.output_tail && (
              <pre className="mono mt-1.5 max-h-40 overflow-y-auto whitespace-pre-wrap rounded bg-surface/60 px-2 py-1.5 text-[11px] leading-snug text-fg-muted">
                {t.output_tail}
              </pre>
            )}
          </div>
        ))}
      </div>
    </Dialog>
  );
}
