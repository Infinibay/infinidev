import { Spinner } from "@harbor/components/display/Spinner";
import { useAppState } from "@/state/store";

const PHASE_LABEL: Record<string, string> = {
  working: "Working",
  chat: "Thinking",
  analysis: "Planning",
  review: "Reviewing",
  idle: "Ready",
};

export function TaskHeader() {
  const s = useAppState();
  if (!s.busy) return null;
  return (
    <div className="flex items-center gap-2 border-b border-fg/8 bg-surface-2/40 px-4 py-2 text-xs">
      <Spinner />
      <span className="font-medium text-accent">{PHASE_LABEL[s.phase] ?? s.phase}</span>
      {s.step && <span className="text-fg-subtle">· step {s.step.n}/{s.step.max}</span>}
    </div>
  );
}
