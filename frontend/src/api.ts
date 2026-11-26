import type {
  BenchmarkReport,
  ChatDetail,
  ChatSummary,
  JudgeSettings,
  McpStatus,
  RunModeType,
  TracePayload,
} from "./types";

async function reqJson<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, init);
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data?.error || `HTTP ${resp.status}`);
  }
  return data as T;
}

export const api = {
  listChats: () => reqJson<{ chats: ChatSummary[] }>("/api/chats"),
  createChat: (title = "New Chat") =>
    reqJson<ChatSummary>("/api/chats", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    }),
  getChat: (chatId: string) => reqJson<ChatDetail>(`/api/chats/${encodeURIComponent(chatId)}`),
  deleteChat: (chatId: string) =>
    reqJson<{ deleted: boolean }>(`/api/chats/${encodeURIComponent(chatId)}`, { method: "DELETE" }),
  sendMessage: (
    chatId: string,
    payload: { prompt: string; mode: RunModeType; attachments: Array<{ name: string; content: string }> }
  ) =>
    reqJson<{ chat_id: string; run_id: string; status: string }>(
      `/api/chats/${encodeURIComponent(chatId)}/messages`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    ),
  cancelRun: (runId: string) =>
    reqJson<{ run_id: string; cancelled: boolean }>(`/api/runs/${encodeURIComponent(runId)}/cancel`, {
      method: "POST",
    }),
  getSettings: () => reqJson<JudgeSettings>("/api/settings"),
  updateSettings: (patch: Record<string, unknown>) =>
    reqJson<JudgeSettings>("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }),
  getMcpStatus: () => reqJson<McpStatus>("/api/mcp/status"),
  listLiveRuns: () =>
    reqJson<{ runs: Array<{ run_id: string; mode: string; status: string; model: string; created_at: string }> }>(
      "/api/live/runs"
    ),
  getLiveRun: (runId: string) => reqJson<TracePayload>(`/api/live/runs/${encodeURIComponent(runId)}`),
  listReplayFiles: () =>
    reqJson<{ files: Array<{ filename: string; run_id: string; status: string; model: string }> }>(
      "/api/replay/files"
    ),
  getReplayFile: (filename: string) =>
    reqJson<TracePayload>(`/api/replay/files/${encodeURIComponent(filename)}`),
  getBenchmark: () => reqJson<BenchmarkReport>("/api/benchmark"),
};
