import React, { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { ChatDetail, ChatMessage, TracePayload } from "../types";
import { acknowledgeSubmission, historicalAnswer, submissionKey } from "../conversation";
import { ConversationComposer, type ChatAttachment } from "./ConversationComposer";
import { AiOfficeWorkspace } from "./AiOfficeWorkspace";
import { MarkdownRenderer } from "./MarkdownRenderer";

interface ChatViewProps {
  chatId: string | null;
  onChatUpdated: () => void;
  onInspectRun: (runId: string) => void;
}

export const ChatView: React.FC<ChatViewProps> = ({
  chatId,
  onChatUpdated,
  onInspectRun,
}) => {
  const [chat, setChat] = useState<ChatDetail | null>(null);
  const [activeTrace, setActiveTrace] = useState<TracePayload | null>(null);
  const [prompt, setPrompt] = useState("");
  const [attachments, setAttachments] = useState<ChatAttachment[]>([]);
  const [sending, setSending] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const sendingLock = useRef(false);
  const [showActivity, setShowActivity] = useState(false);
  const [comparison, setComparison] = useState<{ original: ChatMessage; runId: string } | null>(null);
  const timelineEnd = useRef<HTMLDivElement>(null);

  const activeRunMsg = chat?.messages.find((m) => m.role === "assistant" && m.status === "running") || null;
  const latestAssistantRunId =
    [...(chat?.messages || [])].reverse().find((m) => m.role === "assistant" && m.run_id)?.run_id || null;

  useEffect(() => {
    if (!chatId) {
      setChat(null);
      setActiveTrace(null);
      return;
    }
    setError(null);
    setComparison(null);
    setShowActivity(false);
    setPrompt("");
    setAttachments([]);
    api
      .getChat(chatId)
      .then(setChat)
      .catch((e) => setError(e.message));
  }, [chatId]);

  useEffect(() => {
    if (!latestAssistantRunId || !showActivity) {
      setActiveTrace(null);
      return;
    }
    let cancelled = false;
    api
      .getLiveRun(latestAssistantRunId)
      .then((t) => {
        if (!cancelled) setActiveTrace(t);
      })
      .catch(() => {
        /* run row may still be initializing */
      });
    return () => {
      cancelled = true;
    };
  }, [latestAssistantRunId, showActivity]);

  useEffect(() => {
    if (!chatId || !activeRunMsg) return;
    let cancelled = false;
    let timerId: number | undefined;
    let lastSeq = 0;

    const pollOnce = async () => {
      if (cancelled) return;
      try {
        const updated = await api.getChat(chatId);
        if (cancelled) return;
        setChat(updated);
        const runId = activeRunMsg.run_id;
        if (runId && showActivity) {
          const delta = await api.getLiveRun(runId, lastSeq).catch(() => null);
          if (cancelled) return;
          if (delta) {
            if (typeof delta.latest_seq === "number" && delta.latest_seq > lastSeq) {
              lastSeq = delta.latest_seq;
            }
            setActiveTrace((prev) => {
              if (!prev || prev.run_id !== delta.run_id || !delta.incremental) {
                return delta;
              }
              const merged = [...(prev.execution_events || [])];
              const seen = new Set(
                merged.map((e) => e.event_id || `${e.seq}:${e.event_type}:${e.actor || ""}`)
              );
              for (const ev of delta.execution_events || []) {
                const k = ev.event_id || `${ev.seq}:${ev.event_type}:${ev.actor || ""}`;
                if (!seen.has(k)) {
                  seen.add(k);
                  merged.push(ev);
                }
              }
              merged.sort((a, b) => Number(a.seq || 0) - Number(b.seq || 0));
              return { ...delta, execution_events: merged, incremental: false };
            });
          }
        }
        const stillRunning = updated.messages.some(
          (m) => m.role === "assistant" && m.status === "running"
        );
        if (!stillRunning) {
          onChatUpdated();
          return;
        }
      } catch {
        /* transient poll ignore */
      }
      if (!cancelled) {
        timerId = window.setTimeout(() => {
          void pollOnce();
        }, 650);
      }
    };

    timerId = window.setTimeout(() => {
      void pollOnce();
    }, 350);

    return () => {
      cancelled = true;
      if (timerId !== undefined) window.clearTimeout(timerId);
    };
  }, [chatId, activeRunMsg?.message_id, activeRunMsg?.run_id, onChatUpdated, showActivity]);

  useEffect(() => {
    timelineEnd.current?.scrollIntoView?.({ block: "nearest" });
  }, [chat?.messages.length, activeRunMsg?.status]);

  const handleSend = async () => {
    if (!chatId || !prompt.trim() || sendingLock.current || activeRunMsg) return;
    sendingLock.current = true;
    setSending(true);
    setError(null);
    try {
      const payload = { prompt: prompt.trim(), mode: "network" as const, attachments };
      const request_id = await submissionKey(`private.${chatId}`, payload);
      await api.sendMessage(chatId, { ...payload, request_id });
      acknowledgeSubmission(`private.${chatId}`);
      setPrompt("");
      setAttachments([]);
      const updated = await api.getChat(chatId);
      setChat(updated);
      onChatUpdated();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      sendingLock.current = false;
      setSending(false);
    }
  };

  const handleCancel = async () => {
    if (!activeRunMsg?.run_id) return;
    try {
      await api.cancelRun(activeRunMsg.run_id);
      if (chatId) setChat(await api.getChat(chatId));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const copyMsg = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 1500);
  };

  async function compare(message: ChatMessage) {
    if (!message.run_id || sendingLock.current) return;
    sendingLock.current = true;
    try {
      const runId = message.comparison_run_id ?? (await api.compareRun(message.run_id)).run_id;
      setComparison({ original: message, runId });
    } catch (err) {
      setError((err as Error).message);
    } finally { sendingLock.current = false; }
  }

  return <div className="chat-layout private-conversation">
    <div className="conversation-bar">
      <span>{chat?.title ?? "CrewForge"}</span>
      {latestAssistantRunId && <button type="button" className="public-secondary"
        onClick={() => setShowActivity(!showActivity)} aria-expanded={showActivity}>Agent activity</button>}
    </div>
    {error && <div className="alert-error" role="alert">{error}</div>}
    {comparison ? <PrivateComparison runId={comparison.runId} original={comparison.original}
      onBack={() => setComparison(null)} /> : <>
      <div className="chat-messages conversation-timeline" aria-label="Conversation">
        {(!chat || chat.messages.length === 0) && <div className="conversation-welcome">
          <span aria-hidden="true" className="brand-badge">CF</span>
          <h1>What can we help you with?</h1>
          <p>Ask a question or bring a document. CrewForge assembles a team for each message.</p>
        </div>}
        {chat?.messages.map((message) => <article key={message.message_id}
          className={message.role === "user" ? "conversation-user" : "conversation-answer"}>
          <strong>{message.role === "user" ? "You" : "CrewForge"}</strong>
          {!!message.attachments.length && <div className="att-row">{message.attachments.map((a, i) =>
            <span key={`${a.name}-${i}`} className="att-chip">{a.name}</span>
          )}</div>}
          {message.status === "running" && <p role="status">The team is working…</p>}
          {message.status === "cancelled" && !message.content && <p>Generation stopped.</p>}
          {message.error && <p role="alert">{message.error}</p>}
          {message.content && (message.role === "user" ? <p>{message.content}</p>
            : <MarkdownRenderer content={historicalAnswer(message.content)} />)}
          {message.role === "assistant" && message.content && message.status !== "running" &&
            <div className="guest-actions">
              <button type="button" className="public-secondary"
                onClick={() => copyMsg(message.message_id, historicalAnswer(message.content))}>
                {copiedId === message.message_id ? "Copied" : "Copy"}
              </button>
              {message.run_id && <button type="button" className="public-secondary"
                onClick={() => onInspectRun(message.run_id!)}>Sources & details</button>}
              {message.mode === "network" && ["completed", "partial"].includes(message.status) &&
                <details className="conversation-more"><summary>More</summary>
                  <button type="button" className="public-secondary" onClick={() => void compare(message)}>
                    Compare with single agent
                  </button>
                </details>}
            </div>}
        </article>)}
        <div ref={timelineEnd} />
      </div>
      <ConversationComposer prompt={prompt} setPrompt={setPrompt} attachments={attachments}
        setAttachments={setAttachments} onError={setError} onSend={() => void handleSend()}
        onStop={activeRunMsg ? () => void handleCancel() : undefined}
        disabled={!chatId || chat?.chat_id !== chatId || sending || Boolean(activeRunMsg)} />
    </>}
    {showActivity && <aside className="conversation-activity" aria-label="Agent activity">
      <button type="button" className="public-secondary" onClick={() => setShowActivity(false)}>Close activity</button>
      <AiOfficeWorkspace trace={activeTrace} isRunning={Boolean(activeRunMsg)} mode="network" />
    </aside>}
  </div>;
};

function PrivateComparison({ runId, original, onBack }: {
  runId: string; original: ChatMessage; onBack: () => void;
}) {
  const [single, setSingle] = useState<TracePayload | null>(null);
  const [singleAnswer, setSingleAnswer] = useState<{ status: string; answer: string } | null>(null);
  const [team, setTeam] = useState<TracePayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let open = true;
    let timer: number | undefined;
    async function load() {
      try {
        const reply = await api.getComparison(runId);
        if (!open) return;
        setSingleAnswer(reply);
        if (reply.error) setError(reply.error);
        if (reply.status !== "running") {
          const run = await api.getLiveRun(runId).catch(() => null);
          if (open) setSingle(run);
          return;
        }
      } catch (err) {
        if (open) setError((err as Error).message);
      }
      if (open) timer = window.setTimeout(load, 1200);
    }
    if (original.run_id) void api.getLiveRun(original.run_id).then((run) => {
      if (open) setTeam(run);
    }).catch(() => {});
    void load();
    return () => { open = false; window.clearTimeout(timer); };
  }, [runId, original.run_id]);

  function measurements(run: TracePayload | null) {
    const usage = run?.usage;
    return [
      run?.latency_ms == null ? "—" : `${(run.latency_ms / 1000).toFixed(2)} s`,
      usage?.model_calls ?? "—",
      usage?.usage_reported ? (usage.total_tokens ?? ((usage.input_tokens ?? 0) + (usage.output_tokens ?? 0))) : "—",
      usage?.usage_reported && usage.actual_cost_usd != null ? `$${usage.actual_cost_usd.toFixed(7)}` : "—",
    ];
  }

  return <section className="public-card guest-comparison">
    <button type="button" className="public-secondary" onClick={onBack}>Back to chat</button>
    <h1>Compare answers</h1>
    <div className="guest-results">
      <article><h2>Agent team</h2><MarkdownRenderer content={historicalAnswer(original.content)} /></article>
      <article><h2>Single agent</h2>
        {error && <p role="alert">{error}</p>}
        {singleAnswer?.answer ? <MarkdownRenderer content={singleAnswer.answer} />
          : !singleAnswer || singleAnswer.status === "running"
            ? <p role="status">The single agent is working…
              <button type="button" className="public-secondary" onClick={() => {
                void api.cancelRun(runId).catch((err) => setError(err.message));
              }}>Stop</button>
            </p> : <p>{singleAnswer.status === "cancelled" ? "Generation stopped." : "No answer was returned."}</p>}
      </article>
    </div>
    <table><thead><tr><th scope="col">Metric</th><th scope="col">Agent team</th>
      <th scope="col">Single agent</th></tr></thead>
      <tbody>{["Elapsed time", "Model calls", "Total tokens", "Actual cost"].map((label, idx) =>
        <tr key={label}><th scope="row">{label}</th><td>{measurements(team)[idx]}</td>
          <td>{measurements(single)[idx]}</td></tr>)}</tbody>
    </table>
  </section>;
}
