import { useEffect, useMemo, useRef, useState } from "react";
import { GuestRequestError, guestPost, guestRequest, isActive, officeTrace } from "../guest";
import type { GuestJob } from "../guest";
import type { RunModeType, ThemeMode } from "../types";
import { acknowledgeSubmission, historicalAnswer, submissionKey } from "../conversation";
import { ConversationComposer, type ChatAttachment } from "./ConversationComposer";
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

function Result({ job, mode, showAnswer = true }: {
  job: GuestJob; mode: RunModeType; showAnswer?: boolean;
}) {
  const { status, trace, result } = job;
  const verification = result?.verification;
  const tools = trace?.tool_executions ?? [];
  return (
    <article className="public-card guest-result">
      {showAnswer && <h2>{labels[mode]}</h2>}
      <p role="status" className="guest-stage">{status.status}
        {isActive(job) && status.current_stage !== status.status
          ? ` · ${status.current_stage.replaceAll("_", " ")}` : ""}
      </p>
      {status.error && <p role="alert">{status.error}</p>}
      {showAnswer && result?.answer && <MarkdownRenderer content={
        result.conversational_answer ?? historicalAnswer(result.answer)
      } />}
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

interface GuestChatSummary { chat_id: string; title: string }
interface GuestTurn {
  message_id: string;
  task_id: string;
  prompt: string;
  status: string;
  answer: string;
  error: string | null;
  attachment_names: string[];
  comparison_id: string | null;
}
interface GuestChat extends GuestChatSummary { turns: GuestTurn[] }

function GuestDetails({ taskId }: { taskId: string }) {
  const { job, error } = useJob(taskId);
  return job ? <Result job={job} mode="network" showAnswer={false} />
    : <p role="status">{error ?? "Loading execution details…"}</p>;
}

function GuestActivity({ taskId }: { taskId: string }) {
  const { job, error } = useJob(taskId);
  const trace = useMemo(() => officeTrace(job), [job]);
  return <>{error && <p role="alert">{error}</p>}
    <AiOfficeWorkspace trace={trace} isRunning={job?.status.status === "running"} mode="network" />
    {!!job?.trace?.messages.length && <ul className="activity-contributions">
      {job.trace.messages.map((message) => <li key={message.message_id}>
        <strong>{job.trace?.agents.find((a) => a.agent_id === message.sender_id)?.role
          ?? message.sender_id}</strong>
        <p>{message.content}</p>
      </li>)}
    </ul>}
  </>;
}

function GuestComparison({ teamId, singleId, onBack, onStop }: {
  teamId: string; singleId: string; onBack: () => void; onStop: (taskId: string) => void;
}) {
  const team = useJob(teamId);
  const single = useJob(singleId);
  const left = metrics(team.job);
  const right = metrics(single.job);
  return <section className="public-card guest-comparison">
    <button className="public-secondary" onClick={onBack}>Back to chat</button>
    <h1>Compare answers</h1>
    <p>Both runs use the same message, earlier context, and documents.</p>
    {(team.error || single.error) && <p role="alert">{team.error || single.error}</p>}
    {(!single.job || isActive(single.job)) && <p role="status">The single agent is working…
      <button type="button" className="public-secondary" onClick={() => onStop(singleId)}>Stop</button>
    </p>}
    <div className="guest-results">
      {team.job && <Result job={team.job} mode="network" />}
      {single.job && <Result job={single.job} mode="baseline" />}
    </div>
    <table><thead><tr><th scope="col">Metric</th><th scope="col">Agent team</th><th scope="col">Single agent</th></tr></thead>
      <tbody>{["Elapsed time", "Model calls", "Total tokens", "Actual cost"].map((label, index) =>
        <tr key={label}><th scope="row">{label}</th><td>{left[index]}</td><td>{right[index]}</td></tr>
      )}</tbody>
    </table>
  </section>;
}

export function GuestExperience(props: PublicProps) {
  const [availability, setAvailability] = useState<{ available: boolean; message: string | null } | null>(null);
  const [chats, setChats] = useState<GuestChatSummary[]>([]);
  const [chatId, setChatId] = useState<string | null>(null);
  const [chat, setChat] = useState<GuestChat | null>(null);
  const [prompt, setPrompt] = useState("");
  const [attachments, setAttachments] = useState<ChatAttachment[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [comparison, setComparison] = useState<{ teamId: string; singleId: string } | null>(null);
  const [activity, setActivity] = useState<string | null>(null);
  const [details, setDetails] = useState<string | null>(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);
  const action = useRef(false);
  const end = useRef<HTMLDivElement>(null);
  const currentChat = chat?.chat_id === chatId ? chat : null;
  const turns = currentChat?.turns ?? [];
  const active = turns.find((turn) => ["queued", "running"].includes(turn.status));
  const last = turns.at(-1);

  function select(id: string) {
    sessionStorage.setItem("crewforge.guest.chat", id);
    setChatId(id);
    setActivity(null);
    setDetails(null);
    setComparison(null);
    setHistoryOpen(false);
    setPrompt("");
    setAttachments([]);
    setError(null);
  }

  useEffect(() => {
    const controller = new AbortController();
    async function initialize() {
      try {
        const result = await guestRequest<{ available: boolean; message: string | null }>(
          "/session", { signal: controller.signal },
        );
        const saved = await guestRequest<{ chats: GuestChatSummary[] }>(
          "/chats", { signal: controller.signal },
        );
        if (controller.signal.aborted) return;
        setAvailability(result);
        setChats(saved.chats);
        const preferred = sessionStorage.getItem("crewforge.guest.chat");
        const selection = saved.chats.find((item) => item.chat_id === preferred) ?? saved.chats[0];
        if (selection) select(selection.chat_id);
      } catch (err) {
        if (!controller.signal.aborted) setError((err as Error).message);
      }
    }
    void initialize();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!chatId) return;
    const controller = new AbortController();
    let timer: number | undefined;
    async function load() {
      try {
        const updated = await guestRequest<GuestChat>(`/chats/${chatId}`, { signal: controller.signal });
        if (controller.signal.aborted) return;
        setChat(updated);
        setChats((items) => items.map((item) => item.chat_id === chatId ? updated : item));
        if (updated.turns.some((turn) => ["queued", "running"].includes(turn.status))) {
          timer = window.setTimeout(load, 1300);
        }
      } catch (err) {
        if (!controller.signal.aborted) {
          setError((err as Error).message);
          if (!(err instanceof GuestRequestError) || err.status >= 500) timer = window.setTimeout(load, 3500);
        }
      }
    }
    void load();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [chatId, busy]);

  useEffect(() => {
    end.current?.scrollIntoView?.({ block: "nearest" });
  }, [last?.task_id, last?.status, comparison]);

  async function perform(work: () => Promise<void>) {
    if (action.current) return;
    action.current = true;
    setBusy(true);
    setError(null);
    try { await work(); }
    catch (err) { setError((err as Error).message); }
    finally { action.current = false; setBusy(false); }
  }

  async function create(keepDraft = false): Promise<string> {
    const created = await guestPost<GuestChatSummary>("/chats", {});
    setChats((items) => [created, ...items]);
    if (keepDraft) {
      sessionStorage.setItem("crewforge.guest.chat", created.chat_id);
      setChatId(created.chat_id);
    } else {
      select(created.chat_id);
    }
    return created.chat_id;
  }

  async function send() {
    if (!availability?.available || !prompt.trim() || active) return;
    const payload = {
      prompt: prompt.trim(), documents: attachments.map((file) => ({ title: file.name, content: file.content })),
    };
    await perform(async () => {
      const target = chatId ?? await create(true);
      const request_id = await submissionKey(`guest.${target}`, payload);
      await guestPost(`/chats/${target}/messages`, { ...payload, request_id });
      acknowledgeSubmission(`guest.${target}`);
      setPrompt("");
      setAttachments([]);
      const updated = await guestRequest<GuestChat>(`/chats/${target}`);
      setChat(updated);
    });
  }

  function compare(turn: GuestTurn) {
    if (turn.comparison_id) {
      setComparison({ teamId: turn.task_id, singleId: turn.comparison_id });
      return;
    }
    void perform(async () => {
      const result = await guestPost<{ task_id: string }>("/jobs", { compare_to: turn.task_id });
      setComparison({ teamId: turn.task_id, singleId: result.task_id });
    });
  }

  const stop = (id: string) => { void perform(async () => { await guestPost(`/jobs/${id}/cancel`, {}); }); };

  return <div className="public-shell guest-chat-shell">
    <PublicHeader {...props} />
    <div className="guest-chat-grid">
      <aside className={`guest-history ${historyOpen ? "history-open" : ""}`} aria-label="Conversation history">
        <button type="button" className="public-secondary" disabled={busy || !availability}
          onClick={() => void perform(async () => { await create(); })}>＋ New Chat</button>
        <nav aria-label="Chats">{chats.map((item) => <button key={item.chat_id} type="button"
          className="history-chat" disabled={busy} aria-current={item.chat_id === chatId ? "page" : undefined}
          onClick={() => select(item.chat_id)}>{item.title}</button>)}</nav>
        <a href="#judge-connect">MCP connection guide</a>
      </aside>
      <main className="conversation-main">
        <div className="conversation-bar">
          <button type="button" className="public-secondary" aria-expanded={historyOpen}
            onClick={() => setHistoryOpen(!historyOpen)}>Chats</button>
          <span>{currentChat?.title ?? "CrewForge"}</span>
          {(active ?? last) && <button type="button" className="public-secondary"
            aria-expanded={activity === (active ?? last)?.task_id}
            onClick={() => setActivity(activity ? null : (active ?? last)!.task_id)}>Agent activity</button>}
        </div>
        {comparison ? <GuestComparison {...comparison} onBack={() => setComparison(null)} onStop={stop} /> : <>
          {availability?.message && <p className="guest-availability" role="status">{availability.message}</p>}
          {error && <p className="guest-availability" role="alert">{error}</p>}
          <div className="conversation-timeline" aria-label="Conversation">
            {!turns.length && <div className="conversation-welcome">
              <span aria-hidden="true" className="brand-badge">CF</span>
              <h1>What can we help you with?</h1>
              <p>Ask a question or bring a document. CrewForge assembles a team for each message.</p>
            </div>}
            {turns.map((turn) => <div key={turn.message_id} className="conversation-turn">
              <article className="conversation-user">
                <strong>You</strong><p>{turn.prompt}</p>
                {turn.attachment_names.map((name, i) => <small key={`${name}-${i}`}>{name}</small>)}
              </article>
              <article className="conversation-answer">
                <strong>CrewForge</strong>
                {turn.answer && <MarkdownRenderer content={historicalAnswer(turn.answer)} />}
                {["queued", "running"].includes(turn.status) && <p role="status">
                  {turn.status === "queued" ? "Waiting in the queue…" : "The team is working…"}
                </p>}
                {turn.status === "cancelled" && <p>Generation stopped.</p>}
                {turn.error && <p role="alert">{turn.error}</p>}
                {turn.answer && <div className="guest-actions">
                  <button type="button" className="public-secondary" onClick={async () => {
                    try {
                      await navigator.clipboard.writeText(historicalAnswer(turn.answer));
                      setCopied(turn.task_id);
                    } catch { setError("Couldn’t copy the answer."); }
                  }}>{copied === turn.task_id ? "Copied" : "Copy"}</button>
                  <button type="button" className="public-secondary" aria-expanded={details === turn.task_id}
                    onClick={() => setDetails(details === turn.task_id ? null : turn.task_id)}>Sources & details</button>
                  <details className="conversation-more">
                    <summary>More</summary>
                    <button type="button" className="public-secondary" disabled={busy || (!availability?.available && !turn.comparison_id)}
                      onClick={() => compare(turn)}>Compare with single agent</button>
                    <button type="button" className="public-secondary"
                      onClick={() => setActivity(turn.task_id)}>Agent activity</button>
                  </details>
                </div>}
                {details === turn.task_id && <GuestDetails taskId={turn.task_id} />}
              </article>
            </div>)}
            <div ref={end} />
          </div>
          <ConversationComposer prompt={prompt} setPrompt={setPrompt} attachments={attachments}
            setAttachments={setAttachments} onError={setError} onSend={() => void send()}
            onStop={active ? () => stop(active.task_id) : undefined}
            disabled={busy || !availability?.available || Boolean(active) || (!!chatId && !currentChat)} />
        </>}
        {activity && <aside className="conversation-activity" aria-label="Agent activity">
          <button type="button" className="public-secondary" onClick={() => setActivity(null)}>Close activity</button>
          <GuestActivity taskId={activity} />
        </aside>}
      </main>
    </div>
    <JudgeConnect />
  </div>;
}
