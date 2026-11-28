import { useEffect, useMemo, useState } from "react";
import { GuestRequestError, guestPost, guestRequest, isActive, officeTrace } from "../guest";
import type { GuestJob } from "../guest";
import type { RunModeType, ThemeMode } from "../types";
import { AiOfficeWorkspace } from "./AiOfficeWorkspace";
import { MarkdownRenderer } from "./MarkdownRenderer";
import "./publicExperience.css";

interface PublicProps {
  theme: ThemeMode;
  onToggleTheme: () => void;
  onGuest: () => void;
  onHome: () => void;
  onPrivate: () => void;
}

const labels = { network: "Agent team", baseline: "Single agent" };
const endpoint = () => new URL("/mcp", window.location.origin).href;

function PublicHeader({ theme, onToggleTheme, onHome, onPrivate }: PublicProps) {
  return (
    <header className="public-header">
      <button type="button" className="public-brand" onClick={onHome}>
        <span className="brand-badge" aria-hidden="true">CF</span>
        CrewForge
      </button>
      <nav aria-label="Public navigation">
        <a href="#judge-connect">Connect via MCP</a>
        <button type="button" className="public-secondary" onClick={onToggleTheme}>
          {theme === "dark" ? "Light mode" : "Dark mode"}
        </button>
        <button type="button" className="public-secondary" onClick={onPrivate}>Private workspace</button>
      </nav>
    </header>
  );
}

function JudgeConnect() {
  const [copied, setCopied] = useState(false);
  return (
    <section className="judge-connect public-card" id="judge-connect">
      <div>
        <h2>Connect an AI assistant</h2>
        <p>Add a remote MCP server in your assistant, choose Streamable HTTP, and use this URL. No sign-in is required.</p>
        <div className="judge-endpoint">
          <a href={endpoint()}><code>{endpoint()}</code></a>
          <button type="button" className="public-secondary" onClick={async () => {
            try {
              await navigator.clipboard.writeText(endpoint());
              setCopied(true);
            } catch {
              setCopied(false);
            }
          }}>{copied ? "Copied" : "Copy URL"}</button>
        </div>
        <p>Use <code>run_task</code> to start a team run, then <code>get_task_status</code> and <code>get_task_result</code> to follow it.</p>
      </div>
    </section>
  );
}

export function PublicLanding(props: PublicProps) {
  return (
    <div className="public-shell">
      <PublicHeader {...props} />
      <main className="public-main">
        <section className="public-hero">
          <div>
            <h1>Put an AI team<br /> to work<span>.</span></h1>
            <p>Bring a task to CrewForge and follow the team from assignment through verification. Run the same task with one agent and compare the results.</p>
            <div className="public-entry">
              <button className="public-primary" type="button" onClick={props.onGuest}>Continue as Guest <span aria-hidden="true">→</span></button>
              <small>No account or API key required</small>
            </div>
            <div className="public-entry">
              <button className="public-secondary" type="button" onClick={props.onPrivate}>Log In / Sign Up</button>
              <small>Private workspace with saved chats and your provider settings</small>
            </div>
          </div>
          <div className="public-office-preview">
            <AiOfficeWorkspace trace={null} isRunning={false} mode="network" />
          </div>
        </section>
        <JudgeConnect />
      </main>
    </div>
  );
}

function useJob(taskId: string | null) {
  const [record, setRecord] = useState<{ id: string; job: GuestJob } | null>(null);
  const [failure, setFailure] = useState<{ id: string; message: string } | null>(null);
  useEffect(() => {
    if (!taskId) return;
    const controller = new AbortController();
    let timer: number | undefined;
    const poll = async () => {
      try {
        const job = await guestRequest<GuestJob>(`/jobs/${taskId}`, { signal: controller.signal });
        if (controller.signal.aborted) return;
        setRecord({ id: taskId, job });
        setFailure(null);
        if (isActive(job)) timer = window.setTimeout(poll, 1700);
      } catch (error) {
        if (!controller.signal.aborted) {
          setFailure({ id: taskId, message: (error as Error).message });
          if (!(error instanceof GuestRequestError) || error.status >= 500) {
            timer = window.setTimeout(poll, 5000);
          }
        }
      }
    };
    void poll();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [taskId]);
  return {
    job: record?.id === taskId ? record.job : null,
    error: failure?.id === taskId ? failure.message : null,
  };
}

function Result({ job, mode }: { job: GuestJob; mode: RunModeType }) {
  const { status, trace, result } = job;
  const verification = result?.verification;
  const tools = trace?.tool_executions ?? [];
  return (
    <article className="public-card guest-result">
      <h2>{labels[mode]}</h2>
      <p role="status" className="guest-stage">{status.status}
        {isActive(job) && status.current_stage !== status.status
          ? ` · ${status.current_stage.replaceAll("_", " ")}` : ""}
      </p>
      {status.error && <p role="alert">{status.error}</p>}
      {result?.answer && <MarkdownRenderer content={result.answer} />}
      {verification && (
        <dl className="guest-verification">
          <div><dt>Independent review</dt><dd>{verification.independent_verification_performed ? "Performed" : "Not performed"}</dd></div>
          <div><dt>Honesty Gate</dt><dd>{verification.honesty_gate_passed == null ? "Not applied" : verification.honesty_gate_passed ? "Passed" : "Did not pass"}</dd></div>
          <div><dt>Answer completeness</dt><dd>{verification.answer_complete == null ? "Not established" : verification.answer_complete ? "Complete" : "Incomplete"}</dd></div>
        </dl>
      )}
      {!!result?.unresolved_requirements.length && <details>
        <summary>Unresolved requirements</summary>
        <ul>{result.unresolved_requirements.map((r) =>
          <li key={r.requirement_id}>{r.text}{r.rationale && <p>{r.rationale}</p>}</li>
        )}</ul>
      </details>}
      {!!tools.length && <details>
        <summary>Calculations ({tools.length})</summary>
        <ul>{tools.map((tool) => <li key={tool.tool_execution_id}>
          <code>{tool.expression ?? tool.tool_name}</code>
          {tool.computed_result != null && <> = {tool.computed_result} {tool.units}</>}
          <small>{tool.success ? (tool.inputs_grounded ? "Grounded inputs" : "Inputs not verified") : "Execution failed"}</small>
        </li>)}</ul>
      </details>}
      {!!result?.sources.length && <details>
        <summary>Sources ({result.sources.length})</summary>
        <ul>{result.sources.map((source) => <li key={source.source_ref}>
          {source.title || source.source_ref}
          {result.evidence.filter((e) => e.source_ref === source.source_ref).map((e) =>
            <blockquote key={e.evidence_id}>{e.excerpt}</blockquote>
          )}
        </li>)}</ul>
      </details>}
      {!!trace?.assignments.length && <details>
        <summary>Assignments ({trace.assignments.length})</summary>
        <ul>{trace.assignments.map((task) => <li key={task.task_id}>
          {task.title} · {trace.agents.find((agent) => agent.agent_id === task.assigned_agent_id)?.role ?? task.assigned_agent_id}
          <small>{task.status.replaceAll("_", " ")}</small>
        </li>)}</ul>
      </details>}
      {!!trace?.review_decisions.length && <details>
        <summary>Review decisions ({trace.review_decisions.length})</summary>
        <ul>{trace.review_decisions.map((verdict) => <li key={verdict.verdict_id}>
          <strong>{verdict.decision}</strong> {verdict.rationale}
        </li>)}</ul>
      </details>}
      {!!trace?.messages.length && <details>
        <summary>Team messages ({trace.messages.length})</summary>
        <ul>{trace.messages.map((message) => <li key={message.message_id}>
          <small>{message.sender_id} → {message.recipient_id}</small>
          <p>{message.content}</p>
        </li>)}</ul>
      </details>}
    </article>
  );
}

function elapsed(job: GuestJob | null) {
  if (!job?.status.completed_at) return "—";
  const seconds = (Date.parse(job.status.completed_at) - Date.parse(job.status.created_at)) / 1000;
  return Number.isFinite(seconds) && seconds >= 0 ? `${seconds.toFixed(2)} s` : "—";
}

function metrics(job: GuestJob | null) {
  const usage = job?.result?.usage;
  return [
    elapsed(job),
    usage?.model_calls?.toLocaleString() ?? "—",
    usage?.usage_reported ? usage.total_tokens.toLocaleString() : "—",
    usage?.usage_reported && usage.actual_cost_usd != null ? `$${usage.actual_cost_usd.toFixed(7)}` : "—",
  ];
}

const storageKey = (mode: RunModeType) => `crewforge.guest.${mode}`;

export function GuestExperience(props: PublicProps) {
  const [availability, setAvailability] = useState<{ available: boolean; message: string | null } | null>(null);
  const [prompt, setPrompt] = useState("");
  const [networkId, setNetworkId] = useState(() => sessionStorage.getItem(storageKey("network")));
  const [baselineId, setBaselineId] = useState(() => sessionStorage.getItem(storageKey("baseline")));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const network = useJob(availability ? networkId : null);
  const baseline = useJob(availability ? baselineId : null);
  const activeJob = isActive(baseline.job) ? baseline.job : network.job;
  const waiting = (networkId && !network.job && !network.error) || (baselineId && !baseline.job && !baseline.error);
  const runActive = isActive(network.job) || isActive(baseline.job);
  const trace = useMemo(() => officeTrace(network.job), [network.job]);
  const left = metrics(network.job);
  const right = metrics(baseline.job);

  useEffect(() => {
    const controller = new AbortController();
    guestRequest<{ available: boolean; message: string | null }>("/session", { signal: controller.signal })
      .then(setAvailability)
      .catch((err) => {
        if (!controller.signal.aborted) setAvailability({ available: false, message: err.message });
      });
    return () => controller.abort();
  }, []);

  async function submit(nextMode: RunModeType) {
    if (nextMode === "baseline" && !networkId) return;
    setBusy(true);
    setError(null);
    try {
      const payload: Record<string, string> = nextMode === "network" ? { prompt } : { compare_to: networkId! };
      const { task_id } = await guestPost<{ task_id: string }>("/jobs", payload);
      sessionStorage.setItem(storageKey(nextMode), task_id);
      if (nextMode === "network") {
        setNetworkId(task_id);
        setBaselineId(null);
        sessionStorage.removeItem(storageKey("baseline"));
      } else {
        setBaselineId(task_id);
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function cancel() {
    if (!activeJob) return;
    setBusy(true);
    setError(null);
    try {
      await guestPost(`/jobs/${activeJob.status.task_id}/cancel`, {});
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="public-shell">
      <PublicHeader {...props} />
      <main className="public-main guest-main">
        <h1>Guest office</h1>
        <p>Send a task to CrewForge, follow the team’s work, and compare it with a single agent.</p>
        <div className="guest-availability" role="status">
          {availability ? availability.message || "Guest execution is available." : "Checking guest availability…"}
        </div>
        <form className="public-card guest-form" onSubmit={(event) => {
          event.preventDefault();
          void submit("network");
        }}>
          <label htmlFor="guest-prompt">Your task</label>
          <textarea id="guest-prompt" value={prompt} onChange={(event) => setPrompt(event.target.value)}
            rows={3} maxLength={32000} placeholder="Describe the problem and include any relevant context." />
          <div className="guest-actions">
            <button type="submit" className="public-primary" disabled={!availability?.available || !prompt.trim() || busy || !!waiting || runActive}>Run with agent team</button>
            {network.job && !isActive(network.job) && <button type="button" className="public-secondary"
              disabled={!availability?.available || busy || !!waiting || runActive || !!baselineId}
              onClick={() => void submit("baseline")}>Run same task with single agent</button>}
            {isActive(activeJob) && <button type="button" className="public-secondary" disabled={busy}
              onClick={() => void cancel()}>Cancel run</button>}
          </div>
          {(error || network.error || baseline.error) && <p role="alert">{error || network.error || baseline.error}</p>}
        </form>
        {activeJob?.status.status === "queued" && <p role="status">Waiting for a place in the execution queue.</p>}
        <AiOfficeWorkspace key={networkId ?? "idle"} trace={trace}
          isRunning={network.job?.status.status === "running"} mode="network" />
        {(network.job || baseline.job) && <section className="public-card guest-comparison">
          <h2>Run comparison</h2>
          <table><thead><tr><th scope="col">Metric</th><th scope="col">Agent team</th><th scope="col">Single agent</th></tr></thead>
            <tbody>{["Elapsed time", "Model calls", "Total tokens", "Actual cost"].map((label, index) =>
              <tr key={label}><th scope="row">{label}</th><td>{left[index]}</td><td>{right[index]}</td></tr>
            )}</tbody>
          </table>
        </section>}
        <div className="guest-results">
          {network.job && <Result job={network.job} mode="network" />}
          {baseline.job && <Result job={baseline.job} mode="baseline" />}
        </div>
        <JudgeConnect />
      </main>
    </div>
  );
}
