import { useEffect, useState } from "react";
import { Dialog } from "@harbor/components/overlays/Dialog";
import { Button } from "@harbor/components/buttons/Button";
import { TextField } from "@harbor/components/inputs/TextField";
import { Select } from "@harbor/components/inputs/Select";
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
        <Select
          label="Provider"
          value={provider}
          onChange={setProvider}
          options={providers.map((p) => ({ value: p.id, label: p.display_name }))}
        />
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
