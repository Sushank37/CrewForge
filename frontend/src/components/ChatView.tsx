import React, { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { ChatDetail, RunModeType, TracePayload } from "../types";
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
  const [mode, setMode] = useState<RunModeType>("network");
  const [attachments, setAttachments] = useState<Array<{ name: string; content: string }>>([]);
  const [sending, setSending] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

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
    api
      .getChat(chatId)
      .then(setChat)
      .catch((e) => setError(e.message));
  }, [chatId]);

  useEffect(() => {
    if (!latestAssistantRunId) {
      setActiveTrace(null);
      return;
    }
    api
      .getLiveRun(latestAssistantRunId)
      .then(setActiveTrace)
      .catch(() => {
        /* run row may still be initializing */
      });
  }, [latestAssistantRunId]);

  useEffect(() => {
    if (!chatId || !activeRunMsg) return;
    const timer = setInterval(async () => {
      try {
        const updated = await api.getChat(chatId);
        setChat(updated);
        if (activeRunMsg.run_id) {
          const t = await api.getLiveRun(activeRunMsg.run_id).catch(() => null);
          if (t) setActiveTrace(t);
        }
        const stillRunning = updated.messages.some((m) => m.role === "assistant" && m.status === "running");
        if (!stillRunning) {
          onChatUpdated();
        }
      } catch {
        /* transient poll ignore */
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [chatId, activeRunMsg, onChatUpdated]);

  const handleAttach = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;
    const next: Array<{ name: string; content: string }> = [];
    for (const f of Array.from(files).slice(0, 5)) {
      const content = await f.text();
      next.push({ name: f.name, content });
    }
    setAttachments((prev) => [...prev, ...next].slice(0, 5));
    e.target.value = "";
  };

  const handleSend = async () => {
    if (!chatId || !prompt.trim() || sending || activeRunMsg) return;
    setSending(true);
    setError(null);
    try {
      await api.sendMessage(chatId, { prompt: prompt.trim(), mode, attachments });
      setPrompt("");
      setAttachments([]);
      const updated = await api.getChat(chatId);
      setChat(updated);
      onChatUpdated();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
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

  return (
    <div className="chat-layout">
      <AiOfficeWorkspace
        trace={activeTrace}
        isRunning={Boolean(activeRunMsg)}
        mode={activeRunMsg?.mode || mode}
      />

      {error && <div className="alert-error">{error}</div>}

      <div className="chat-messages">
        {(!chat || chat.messages.length === 0) && (
          <div className="chat-empty">
            <h2>Ask CrewForge anything</h2>
            <p className="muted">
              Submit a question or attach a reference document. Your Lead Orchestrator will assemble
              specialists, verify claims, and return a grounded answer.
            </p>
          </div>
        )}

        {chat?.messages.map((m) => (
          <div key={m.message_id} className={`msg-row msg-${m.role}`}>
            <div className={`msg-bubble msg-bubble-${m.role}`}>
              <div className="msg-meta">
                <strong>{m.role === "user" ? "You" : "CrewForge"}</strong>
                <span className="muted">
                  {m.mode === "baseline" ? "Baseline" : "Agent Network"} · {m.status}
                </span>
              </div>

              {m.attachments.length > 0 && (
                <div className="att-row">
                  {m.attachments.map((a) => (
                    <span key={a.name} className="att-chip">
                      📎 {a.name}
                    </span>
                  ))}
                </div>
              )}

              {m.status === "running" && (
                <div className="muted">Team is working... watch the AI Office above.</div>
              )}

              {m.error && <div className="alert-error">{m.error}</div>}

              {m.content && <MarkdownRenderer content={m.content} />}

              {m.role === "assistant" && m.status !== "running" && (
                <div className="msg-actions">
                  {m.content && (
                    <button
                      type="button"
                      className="btn-ghost btn-xs"
                      onClick={() => copyMsg(m.message_id, m.content)}
                    >
                      {copiedId === m.message_id ? "Copied!" : "Copy"}
                    </button>
                  )}
                  {m.run_id && (
                    <button
                      type="button"
                      className="btn-ghost btn-xs"
                      onClick={() => onInspectRun(m.run_id!)}
                    >
                      Inspect in Mission Control →
                    </button>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="composer-wrap">
        {attachments.length > 0 && (
          <div className="att-row" style={{ marginBottom: 6 }}>
            {attachments.map((a, idx) => (
              <span key={`${a.name}-${idx}`} className="att-chip">
                📎 {a.name}{" "}
                <button
                  type="button"
                  className="att-rm"
                  onClick={() => setAttachments((prev) => prev.filter((_, i) => i !== idx))}
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        )}

        <div className="composer-box">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".md,.txt,.csv,.json"
            className="hidden"
            onChange={handleAttach}
          />
          <button
            type="button"
            className="btn-icon"
            title="Attach reference document"
            onClick={() => fileInputRef.current?.click()}
          >
            📎
          </button>
          <textarea
            rows={2}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder="Ask CrewForge anything..."
          />
          <select
            aria-label="Execution mode"
            value={mode}
            onChange={(e) => setMode(e.target.value as RunModeType)}
            className="select-compact"
          >
            <option value="network">Network</option>
            <option value="baseline">Baseline</option>
          </select>
          {activeRunMsg ? (
            <button type="button" className="btn-danger btn-sm" onClick={handleCancel}>
              Stop
            </button>
          ) : (
            <button
              type="button"
              className="btn-primary btn-sm"
              disabled={sending || !prompt.trim()}
              onClick={handleSend}
            >
              Send
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
