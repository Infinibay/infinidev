import { useEffect, useRef, useState } from "react";
import { Dialog } from "@harbor/components/overlays/Dialog";
import { Button } from "@harbor/components/buttons/Button";
import { TextField } from "@harbor/components/inputs/TextField";
import { Z } from "@harbor/lib/z";
import { useToast } from "@harbor/components/feedback/Toast";
import { useEngine } from "@/state/store";
import type { ProviderDto } from "@/api/engineEvent";

export function SettingsModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { commands, setModel } = useEngine();
  const [providers, setProviders] = useState<ProviderDto[]>([]);
  const [provider, setProvider] = useState("ollama");
  const [model, setModelState] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [temperature, setTemperature] = useState(0.2);
  const [maxIterations, setMaxIterations] = useState(25);
  const [planning, setPlanning] = useState(true);
  const [orchestrate, setOrchestrate] = useState(true);
  const [review, setReview] = useState(true);
  const [forceManualTools, setForceManualTools] = useState(false);
  const [confirmCommands, setConfirmCommands] = useState(false);
  const [confirmWrites, setConfirmWrites] = useState(false);
  const [models, setModels] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const { push } = useToast();

  useEffect(() => {
    if (!open) return;
    (async () => {
      const [provs, cfg] = await Promise.all([commands.providers(), commands.getConfig()]);
      setProviders(provs);
      setProvider(cfg.provider);
      setModelState(cfg.model);
      setApiKey(cfg.api_key ?? "");
      setBaseUrl(cfg.base_url);
      setTemperature(cfg.temperature ?? 0.2);
      setMaxIterations(cfg.max_iterations);
      setPlanning(cfg.planning);
      setOrchestrate(cfg.orchestrate);
      setReview(cfg.review);
      setForceManualTools(cfg.force_manual_tools);
      setConfirmCommands(cfg.confirm_commands);
      setConfirmWrites(cfg.confirm_writes);
    })().catch(() => undefined);
  }, [open, commands]);

  const refresh = async () => {
    setLoading(true);
    try {
      setModels(await commands.listModels(provider, apiKey || null, baseUrl));
    } catch (e) {
      push({ title: "Could not list models", description: String((e as Error).message), tone: "danger" });
    } finally {
      setLoading(false);
    }
  };

  const save = async () => {
    try {
      const cfg = await commands.setConfig({
        provider,
        model,
        api_key: apiKey || null,
        base_url: baseUrl,
        temperature,
        max_iterations: maxIterations,
        planning,
        orchestrate,
        review,
        force_manual_tools: forceManualTools,
        confirm_commands: confirmCommands,
        confirm_writes: confirmWrites,
      });
      setModel(cfg.model);
      push({ title: "Settings saved", tone: "success" });
      onClose();
    } catch (e) {
      push({ title: "Save failed", description: String((e as Error).message), tone: "danger" });
    }
  };

  const prov = providers.find((p) => p.id === provider);

  return (
    <Dialog
      open={open}
      onClose={onClose}
      size="md"
      title="Model & settings"
      footer={
        <div className="ml-auto flex gap-2">
          <Button variant="ghost" onClick={onClose}>
            Close
          </Button>
          <Button variant="primary" onClick={() => void save()}>
            Save
          </Button>
        </div>
      }
    >
      <div className="space-y-3">
        <ProviderSelect providers={providers} value={provider} onChange={setProvider} />
        {prov?.api_key_required && (
          <TextField label="API key" type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} />
        )}
        {prov?.base_url_editable && (
          <TextField
            label="Base URL"
            value={baseUrl}
            placeholder={prov.default_base_url}
            onChange={(e) => setBaseUrl(e.target.value)}
          />
        )}
        <div>
          <div className="mb-1 flex items-center justify-between">
            <span className="text-xs text-fg-muted">Model</span>
            <button onClick={() => void refresh()} className="text-[10px] text-accent">
              {loading ? "loading…" : "refresh list"}
            </button>
          </div>
          <TextField value={model} placeholder="model id" onChange={(e) => setModelState(e.target.value)} />
          {models.length > 0 && (
            <div className="mt-1 max-h-40 overflow-y-auto rounded-md border border-fg/10">
              {models.map((m) => (
                <button
                  key={m}
                  onClick={() => setModelState(m)}
                  className={`mono block w-full px-2 py-1 text-left text-[11px] hover:bg-surface-2/60 ${
                    m === model ? "text-accent" : "text-fg-muted"
                  }`}
                >
                  {m}
                </button>
              ))}
            </div>
          )}
        </div>
        <div className="grid grid-cols-2 gap-3">
          <TextField
            label="Temperature"
            type="number"
            value={String(temperature)}
            onChange={(e) => setTemperature(Number(e.target.value))}
          />
          <TextField
            label="Max iterations"
            type="number"
            value={String(maxIterations)}
            onChange={(e) => setMaxIterations(Math.max(1, Number(e.target.value) || 1))}
          />
        </div>

        <div className="space-y-1.5 border-t border-fg/8 pt-3">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-fg-subtle">
            Engine behavior
          </div>
          <Toggle
            label="Planning"
            hint="Plan the task before executing (plan → execute → summarize)."
            checked={planning}
            onChange={setPlanning}
          />
          <Toggle
            label="Orchestrate (chat tier)"
            hint="Triage each turn respond-vs-escalate before any developer work."
            checked={orchestrate}
            onChange={setOrchestrate}
          />
          <Toggle
            label="Review changes"
            hint="Run a critic pass after the agent edits files."
            checked={review}
            onChange={setReview}
          />
          <Toggle
            label="Force manual tool-calling"
            hint="Use prompt-based tool calls even when the model advertises native ones."
            checked={forceManualTools}
            onChange={setForceManualTools}
          />
          <Toggle
            label="Confirm shell commands"
            hint="Ask before the agent runs a shell command, and let you allow or deny each one."
            checked={confirmCommands}
            onChange={setConfirmCommands}
          />
          <Toggle
            label="Confirm file changes"
            hint="Ask before the agent creates, edits, or deletes a file."
            checked={confirmWrites}
            onChange={setConfirmWrites}
          />
        </div>
      </div>
    </Dialog>
  );
}

/** Provider chooser. A self-contained dropdown rendered *inside* the dialog's
 *  stacking context (not portaled to <body>), so its options sit above the
 *  sibling form fields without fighting the global layer scale — Harbor's
 *  portaled Select would land at Z.POPOVER (1000), below Z.DIALOG (5000), and
 *  be occluded by the modal. The options menu uses Z.RAISED from the layer
 *  system rather than a hardcoded z-index. */
function ProviderSelect({
  providers,
  value,
  onChange,
}: {
  providers: ProviderDto[];
  value: string;
  onChange: (id: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const current = providers.find((p) => p.id === value);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  return (
    <div ref={ref} className="relative">
      <label className="mb-1.5 block text-xs text-fg-muted">Provider</label>
      <button
        type="button"
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between rounded-md border border-fg/15 bg-surface-2/60 px-3 py-2 text-left text-sm text-fg outline-none transition-colors hover:bg-surface-2 focus-visible:border-accent/50"
      >
        <span className={current ? "text-fg" : "text-fg-subtle"}>
          {current?.display_name ?? "Select a provider…"}
        </span>
        <span className={`text-fg-subtle transition-transform ${open ? "rotate-180" : ""}`}>▾</span>
      </button>
      {open && (
        <ul
          role="listbox"
          style={{ zIndex: Z.RAISED }}
          className="absolute left-0 right-0 top-full mt-1 max-h-56 overflow-y-auto rounded-md border border-fg/15 bg-surface-2 shadow-harbor-lg"
        >
          {providers.map((p) => (
            <li key={p.id} role="option" aria-selected={p.id === value}>
              <button
                type="button"
                onClick={() => {
                  onChange(p.id);
                  setOpen(false);
                }}
                className={`block w-full px-3 py-1.5 text-left text-sm transition-colors hover:bg-surface-3/60 ${
                  p.id === value ? "text-accent" : "text-fg-muted"
                }`}
              >
                {p.display_name}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** A compact labelled toggle row (Harbor-styled checkbox). */
function Toggle({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string;
  hint: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-start gap-2.5 rounded-md px-1 py-1 hover:bg-surface-2/40">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        style={{ accentColor: "var(--harbor-color-accent, #8b5cf6)" }}
        className="mt-0.5 h-4 w-4 shrink-0"
      />
      <span className="min-w-0">
        <span className="block text-[13px] text-fg">{label}</span>
        <span className="block text-[11px] leading-snug text-fg-subtle">{hint}</span>
      </span>
    </label>
  );
}
