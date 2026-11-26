import React, { useEffect, useState } from "react";
import { api } from "../api";
import type { JudgeSettings, ThemeMode } from "../types";

interface SettingsViewProps {
  theme: ThemeMode;
  onThemeChange: (t: ThemeMode) => void;
  reducedMotion: boolean;
  onReducedMotionChange: (v: boolean) => void;
}

export const SettingsView: React.FC<SettingsViewProps> = ({
  theme,
  onThemeChange,
  reducedMotion,
  onReducedMotionChange,
}) => {
  const [settings, setSettings] = useState<JudgeSettings | null>(null);
  const [apiKeyInput, setApiKeyInput] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api
      .getSettings()
      .then(setSettings)
      .catch((e) => setErr(e.message));
  }, []);

  const save = async (patch: Record<string, unknown>) => {
    setSaving(true);
    setMsg(null);
    setErr(null);
    try {
      const updated = await api.updateSettings(patch);
      setSettings(updated);
      setApiKeyInput("");
      setMsg("Settings saved to isolated judge session.");
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  if (!settings) {
    return (
      <div className="page-container">
        {err ? <div className="alert-error">{err}</div> : <div className="muted">Loading settings...</div>}
      </div>
    );
  }

  return (
    <div className="page-container">
      <div className="page-header">
        <div>
          <h1>Settings</h1>
          <p className="muted">
            Configure your own provider, API key, model, and safety limits. API keys stay strictly in backend
            memory and are never written to browser storage or source control.
          </p>
        </div>
      </div>

      {msg && <div className="alert-success">{msg}</div>}
      {err && <div className="alert-error">{err}</div>}

      <div className="settings-grid">
        {/* Model & API */}
        <section className="card">
          <h2>Model &amp; API Configuration</h2>

          <div className="field">
            <span>Billing Mode</span>
            <div className="input-row">
              <button
                type="button"
                className={settings.billing_mode === "free" ? "btn-primary btn-sm" : "btn-ghost btn-sm"}
                onClick={() =>
                  setSettings({
                    ...settings,
                    billing_mode: "free",
                    allow_paid_runs: false,
                    max_cost_per_run_usd: 0,
                    total_budget_ceiling_usd: 0,
                    custom_input_price_per_1m: 0,
                    custom_output_price_per_1m: 0,
                    ...(settings.provider === "openai"
                      ? {
                          provider: "openai_compatible",
                          openai_base_url: "https://api.groq.com/openai/v1",
                          model: "openai/gpt-oss-120b",
                        }
                      : {}),
                  })
                }
              >
                Free Mode ($0.00 Limit)
              </button>
              <button
                type="button"
                className={settings.billing_mode === "byok" ? "btn-primary btn-sm" : "btn-ghost btn-sm"}
                onClick={() =>
                  setSettings({
                    ...settings,
                    billing_mode: "byok",
                    max_cost_per_run_usd: settings.max_cost_per_run_usd || 0.05,
                    total_budget_ceiling_usd: settings.total_budget_ceiling_usd || 5.0,
                  })
                }
              >
                Bring Your Own API Key (BYOK)
              </button>
            </div>
          </div>

          {settings.billing_mode === "free" && (
            <div className="input-row">
              <button
                type="button"
                className="btn-ghost btn-sm"
                onClick={() =>
                  setSettings({
                    ...settings,
                    provider: "openai_compatible",
                    openai_base_url: "https://api.groq.com/openai/v1",
                    model: "openai/gpt-oss-120b",
                  })
                }
              >
                Preset: Groq Free (gpt-oss-120b)
              </button>
              <button
                type="button"
                className="btn-ghost btn-sm"
                onClick={() =>
                  setSettings({
                    ...settings,
                    provider: "openai_compatible",
                    openai_base_url: "https://api.groq.com/openai/v1",
                    model: "openai/gpt-oss-20b",
                  })
                }
              >
                Preset: Groq Alt (gpt-oss-20b)
              </button>
            </div>
          )}

          <label className="field">
            <span>Provider</span>
            <select
              value={settings.provider}
              onChange={(e) =>
                setSettings({ ...settings, provider: e.target.value as JudgeSettings["provider"] })
              }
            >
              <option value="ollama">Ollama (Local)</option>
              {settings.billing_mode === "byok" && <option value="openai">OpenAI</option>}
              <option value="openai_compatible">
                {settings.billing_mode === "free" ? "Groq Free-Tier (OpenAI-Compatible)" : "OpenAI-Compatible Endpoint"}
              </option>
            </select>
          </label>

          {settings.billing_mode === "free" && settings.provider === "openai_compatible" && (
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={settings.free_tier_verified}
                onChange={(e) => setSettings({ ...settings, free_tier_verified: e.target.checked })}
              />
              <span>Confirm this is a verified Groq free-tier account ($0.00 spending cap)</span>
            </label>
          )}

          <label className="field">
            <span>Model</span>
            <input
              type="text"
              value={settings.model}
              onChange={(e) => setSettings({ ...settings, model: e.target.value })}
              placeholder="e.g. qwen3:4b or gpt-6-luna"
            />
          </label>

          {settings.provider !== "ollama" && (
            <label className="field">
              <span>
                API Key{" "}
                <em className="muted">
                  ({settings.api_key_configured ? `Configured (${settings.api_key_hint})` : "Not configured"})
                </em>
              </span>
              <div className="input-row">
                <input
                  type={showKey ? "text" : "password"}
                  value={apiKeyInput}
                  onChange={(e) => setApiKeyInput(e.target.value)}
                  placeholder="Enter API key (stored in server RAM only)"
                  autoComplete="off"
                />
                <button type="button" className="btn-ghost btn-sm" onClick={() => setShowKey((s) => !s)}>
                  {showKey ? "Hide" : "Show"}
                </button>
                {settings.api_key_configured && (
                  <button
                    type="button"
                    className="btn-ghost btn-sm"
                    onClick={() => save({ clear_api_key: true })}
                  >
                    Clear
                  </button>
                )}
              </div>
            </label>
          )}

          {settings.provider === "openai_compatible" && (
            <label className="field">
              <span>Compatible Base URL</span>
              <input
                type="url"
                value={settings.openai_base_url}
                onChange={(e) => setSettings({ ...settings, openai_base_url: e.target.value })}
                placeholder="https://api.example.com/v1"
              />
            </label>
          )}

          {settings.provider === "ollama" && (
            <label className="field">
              <span>Ollama Base URL</span>
              <input
                type="url"
                value={settings.ollama_base_url}
                onChange={(e) => setSettings({ ...settings, ollama_base_url: e.target.value })}
              />
            </label>
          )}

          <label className="field">
            <span>Reasoning Effort (Cloud)</span>
            <select
              value={settings.openai_reasoning_effort}
              onChange={(e) =>
                setSettings({
                  ...settings,
                  openai_reasoning_effort: e.target.value as JudgeSettings["openai_reasoning_effort"],
                })
              }
            >
              <option value="none">none</option>
              <option value="low">low</option>
              <option value="medium">medium</option>
              <option value="high">high</option>
            </select>
          </label>
        </section>

        {/* Execution Limits & Judge Session Isolation */}
        <section className="card">
          <h2>Execution Limits &amp; Safety</h2>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={settings.allow_paid_runs}
              onChange={(e) => setSettings({ ...settings, allow_paid_runs: e.target.checked })}
            />
            <span>Explicitly authorize paid cloud execution (<code>allow_paid_runs</code>)</span>
          </label>

          <div className="field-2">
            <label className="field">
              <span>Max Output Tokens</span>
              <input
                type="number"
                value={settings.max_output_tokens}
                onChange={(e) => setSettings({ ...settings, max_output_tokens: Number(e.target.value) })}
              />
            </label>
            <label className="field">
              <span>Max Model Calls / Run</span>
              <input
                type="number"
                value={settings.max_model_calls_per_run}
                onChange={(e) => setSettings({ ...settings, max_model_calls_per_run: Number(e.target.value) })}
              />
            </label>
            <label className="field">
              <span>Max Workers</span>
              <input
                type="number"
                value={settings.max_workers}
                onChange={(e) => setSettings({ ...settings, max_workers: Number(e.target.value) })}
              />
            </label>
            <label className="field">
              <span>Max Tasks</span>
              <input
                type="number"
                value={settings.max_tasks}
                onChange={(e) => setSettings({ ...settings, max_tasks: Number(e.target.value) })}
              />
            </label>
            <label className="field">
              <span>Max Cost / Run (USD)</span>
              <input
                type="number"
                step="0.005"
                value={settings.max_cost_per_run_usd}
                onChange={(e) => setSettings({ ...settings, max_cost_per_run_usd: Number(e.target.value) })}
              />
            </label>
            <label className="field">
              <span>Session Budget Cap (USD)</span>
              <input
                type="number"
                step="0.5"
                value={settings.total_budget_ceiling_usd}
                onChange={(e) => setSettings({ ...settings, total_budget_ceiling_usd: Number(e.target.value) })}
              />
            </label>
            <label className="field">
              <span>Custom Input Price / 1M (USD)</span>
              <input
                type="number"
                step="0.01"
                placeholder="Optional (e.g. 0.10)"
                value={settings.custom_input_price_per_1m ?? ""}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    custom_input_price_per_1m: e.target.value === "" ? null : Number(e.target.value),
                  })
                }
              />
            </label>
            <label className="field">
              <span>Custom Output Price / 1M (USD)</span>
              <input
                type="number"
                step="0.01"
                placeholder="Optional (e.g. 0.50)"
                value={settings.custom_output_price_per_1m ?? ""}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    custom_output_price_per_1m: e.target.value === "" ? null : Number(e.target.value),
                  })
                }
              />
            </label>
          </div>

          <div className="muted" style={{ fontSize: 12 }}>
            Isolated trace DB: <code>{settings.session_db_path}</code>
            <br />
            Isolated budget DB: <code>{settings.session_budget_db_path}</code>
          </div>
        </section>

        {/* Appearance */}
        <section className="card">
          <h2>Appearance &amp; Accessibility</h2>
          <div className="field">
            <span>Theme</span>
            <div className="input-row">
              <button
                type="button"
                className={theme === "dark" ? "btn-primary btn-sm" : "btn-ghost btn-sm"}
                onClick={() => onThemeChange("dark")}
              >
                Dark Mode
              </button>
              <button
                type="button"
                className={theme === "light" ? "btn-primary btn-sm" : "btn-ghost btn-sm"}
                onClick={() => onThemeChange("light")}
              >
                Light Mode
              </button>
            </div>
          </div>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={reducedMotion}
              onChange={(e) => onReducedMotionChange(e.target.checked)}
            />
            <span>Reduce office character &amp; SVG animations</span>
          </label>
        </section>
      </div>

      <div style={{ marginTop: 16 }}>
        <button
          type="button"
          className="btn-primary"
          disabled={saving}
          onClick={() =>
            save({
              billing_mode: settings.billing_mode,
              free_tier_verified: settings.free_tier_verified,
              provider: settings.provider,
              model: settings.model,
              openai_base_url: settings.openai_base_url,
              ollama_base_url: settings.ollama_base_url,
              openai_reasoning_effort: settings.openai_reasoning_effort,
              allow_paid_runs: settings.allow_paid_runs,
              max_output_tokens: settings.max_output_tokens,
              max_workers: settings.max_workers,
              max_tasks: settings.max_tasks,
              max_concurrency: settings.max_concurrency,
              max_model_calls_per_run: settings.max_model_calls_per_run,
              max_cost_per_run_usd: settings.max_cost_per_run_usd,
              total_budget_ceiling_usd: settings.total_budget_ceiling_usd,
              custom_input_price_per_1m: settings.custom_input_price_per_1m,
              custom_output_price_per_1m: settings.custom_output_price_per_1m,
              ...(apiKeyInput.trim() ? { api_key: apiKeyInput.trim() } : {}),
            })
          }
        >
          {saving ? "Saving..." : "Save Configuration"}
        </button>
      </div>
    </div>
  );
};
