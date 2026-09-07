import { useEffect, useState } from "react";
import {
  Brain,
  ChevronDown,
  Cpu,
  KeyRound,
  RefreshCw,
  Settings2,
} from "lucide-react";
import { api } from "./api";
import { ErrorNotice, Loading, PageTitle, SaveButton } from "./components";
import { useResource } from "./hooks";
import type { Models } from "./types";
interface SettingsData {
  values: Record<string, string | boolean | number>;
  secrets: Record<string, boolean>;
}
export function SettingsView({ onSaved }: { onSaved: () => void }) {
  const resource = useResource<SettingsData>("/settings");
  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");
  const [effort, setEffort] = useState("");
  const [key, setKey] = useState("");
  const [base, setBase] = useState("");
  const [workers, setWorkers] = useState(3);
  const [iterations, setIterations] = useState(50);
  const [tokens, setTokens] = useState(4096);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");
  const [discover, setDiscover] = useState(false);
  const models = useResource<Models>(
    provider
      ? `/models?provider=${encodeURIComponent(provider)}&model=${encodeURIComponent(model)}&refresh=${discover}`
      : "/models",
  );
  useEffect(() => {
    if (resource.data) {
      const v = resource.data.values;
      setProvider(String(v.LLM_PROVIDER));
      setModel(String(v.LLM_MODEL).split("/").pop() || "");
      setBase(String(v.LLM_BASE_URL));
      setWorkers(Number(v.TEAM_MAX_WORKERS));
      setIterations(Number(v.LOOP_MAX_ITERATIONS));
      setTokens(Number(v.THINKING_BUDGET_TOKENS));
    }
  }, [resource.data]);
  useEffect(() => {
    if (models.data) setEffort(models.data.effort.value);
  }, [models.data]);
  const selected = models.data?.providers.find((p) => p.id === provider);
  function changeProvider(value: string) {
    const config = models.data?.providers.find((p) => p.id === value);
    setProvider(value);
    setModel(config?.static_models[0] || "");
    setBase(config?.default_base_url || "");
    setKey("");
    setDiscover(false);
    setSaved(false);
  }
  async function save() {
    setSaving(true);
    setError("");
    setSaved(false);
    const updates: Record<string, unknown> = {
      LLM_PROVIDER: provider,
      LLM_MODEL: model,
      LLM_BASE_URL: base,
      TEAM_MAX_WORKERS: workers,
      LOOP_MAX_ITERATIONS: iterations,
      THINKING_BUDGET_TOKENS: tokens,
    };
    if (effort && models.data?.effort.choices.includes(effort))
      updates.THINKING_BUDGET = effort;
    if (key) updates.LLM_API_KEY = key;
    try {
      await api("/settings", { updates }, "PATCH");
      setKey("");
      setSaved(true);
      resource.reload();
      onSaved();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  }
  return (
    <div className="page-scroll settings-page">
      <PageTitle
        eyebrow="Your harness"
        title="Set up the way you work."
        actions={
          <SaveButton
            busy={saving}
            saved={saved}
            disabled={!resource.data || models.loading}
            onClick={() => void save()}
          />
        }
      >
        Choose a model and its supported reasoning controls. Changes apply to
        subsequent turns.
      </PageTitle>
      <ErrorNotice error={error || resource.error} />
      {resource.loading ? (
        <Loading />
      ) : (
        <div className="settings-sections">
          <section className="settings-card">
            <div className="settings-card-title">
              <Cpu size={20} />
              <div>
                <h3>Model connection</h3>
                <p>
                  The principal agent and its specialists use this configured
                  connection.
                </p>
              </div>
            </div>
            <div className="form-grid">
              <label>
                Provider
                <div className="select-wrap">
                  <select
                    value={provider}
                    onChange={(e) => changeProvider(e.target.value)}
                  >
                    {models.data?.providers.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.display_name}
                      </option>
                    ))}
                  </select>
                  <ChevronDown size={15} />
                </div>
              </label>
              <label>
                Model
                <input
                  list="models"
                  value={model}
                  onChange={(e) => {
                    setModel(e.target.value);
                    setSaved(false);
                    setDiscover(false);
                  }}
                  placeholder="Model identifier"
                />
                <datalist id="models">
                  {models.data?.models.map((value) => (
                    <option value={value} key={value} />
                  ))}
                </datalist>
              </label>
            </div>
            <div className="model-discovery">
              <span>
                Use the provider’s model identifier, or select a known model.
              </span>
              <button
                className="text-button"
                disabled={models.loading}
                onClick={() => {
                  setDiscover(true);
                  models.reload();
                }}
              >
                <RefreshCw size={13} />
                Discover models
              </button>
            </div>
            <ErrorNotice error={models.error || models.data?.error || ""} />
            {selected?.base_url_editable && (
              <label className="form-field">
                API base URL
                <input value={base} onChange={(e) => setBase(e.target.value)} />
              </label>
            )}
            {selected?.api_key_required && (
              <label className="form-field">
                <span>
                  <KeyRound size={14} /> API key
                </span>
                <input
                  type="password"
                  autoComplete="new-password"
                  value={key}
                  onChange={(e) => setKey(e.target.value)}
                  placeholder={
                    provider === resource.data?.values.LLM_PROVIDER &&
                    resource.data?.secrets.LLM_API_KEY
                      ? "Configured · leave blank to keep it"
                      : "Enter an API key"
                  }
                />
                <small>
                  Stored by Infinidev on this computer. Existing keys are never
                  returned to the browser.
                </small>
              </label>
            )}
            {provider.endsWith("_subscription") && (
              <p className="information-note">
                Uses credentials from the provider’s existing CLI login on this
                computer. Complete that login in your terminal.
              </p>
            )}
          </section>
          <section className="settings-card">
            <div className="settings-card-title">
              <Brain size={20} />
              <div>
                <h3>Reasoning effort</h3>
                <p>
                  {models.data?.effort.description ||
                    "Checking the selected model…"}
                </p>
              </div>
            </div>
            {models.data?.effort.choices.length ? (
              <>
                <div
                  className="effort-options"
                  role="group"
                  aria-label="Reasoning effort"
                >
                  {models.data.effort.choices.map((value) => (
                    <button
                      className={effort === value ? "selected" : ""}
                      aria-pressed={effort === value}
                      key={value}
                      onClick={() => {
                        setEffort(value);
                        setSaved(false);
                      }}
                    >
                      {value}
                    </button>
                  ))}
                </div>
                {effort === "custom" && (
                  <label className="form-field">
                    Thinking token budget
                    <input
                      type="number"
                      min="1"
                      max="1000000"
                      value={tokens}
                      onChange={(e) => setTokens(Number(e.target.value))}
                    />
                  </label>
                )}
              </>
            ) : (
              <p className="muted">
                This model and API do not advertise a verified effort control.
              </p>
            )}
          </section>
          <section className="settings-card">
            <div className="settings-card-title">
              <Settings2 size={20} />
              <div>
                <h3>Execution</h3>
                <p>
                  The orchestrator decides when to delegate within these limits.
                </p>
              </div>
            </div>
            <div className="form-grid">
              <label>
                Parallel specialists
                <input
                  type="number"
                  min="1"
                  max="16"
                  value={workers}
                  onChange={(e) => {
                    setWorkers(Number(e.target.value));
                    setSaved(false);
                  }}
                />
              </label>
              <label>
                Developer iteration limit
                <input
                  type="number"
                  min="1"
                  max="1000000"
                  value={iterations}
                  onChange={(e) => {
                    setIterations(Number(e.target.value));
                    setSaved(false);
                  }}
                />
              </label>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
