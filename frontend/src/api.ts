import type {
  AuthMeResponse,
  AuthUser,
  BenchmarkReport,
  ChatDetail,
  ChatSummary,
  JudgeSettings,
  McpStatus,
  RunModeType,
  TracePayload,
} from "./types";

export function getSessionToken(): string | null {
  try {
    return window.sessionStorage.getItem("crewforge_token") || null;
  } catch {
    return null;
  }
}

export function setSessionToken(token: string | null): void {
  try {
    if (token && token.trim()) {
      window.sessionStorage.setItem("crewforge_token", token.trim());
    } else {
      window.sessionStorage.removeItem("crewforge_token");
    }
  } catch {
    // Ignore storage errors in restricted contexts
  }
}

function getCsrfToken(): string | null {
  try {
    const match = document.cookie.match(/(?:^|;\s*)crewforge_csrf=([^;]+)/);
    return match ? decodeURIComponent(match[1]) : null;
  } catch {
    return null;
  }
}

async function reqJson<T>(url: string, init?: RequestInit): Promise<T> {
  const token = getSessionToken();
  const headers = new Headers(init?.headers || {});
  if (token) headers.set("X-CrewForge-Token", token);
  const method = (init?.method || "GET").toUpperCase();
  if (["POST", "PUT", "PATCH", "DELETE"].includes(method)) {
    const csrf = getCsrfToken();
    if (csrf) headers.set("X-CSRF-Token", csrf);
  }
  const resp = await fetch(url, { ...init, headers, credentials: "same-origin" });
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
  getLiveRun: (runId: string, sinceSeq?: number) =>
    reqJson<TracePayload>(
      `/api/live/runs/${encodeURIComponent(runId)}${
        sinceSeq && sinceSeq > 0 ? `?since_seq=${sinceSeq}` : ""
      }`
    ),
  listReplayFiles: () =>
    reqJson<{ files: Array<{ filename: string; run_id: string; status: string; model: string }> }>(
      "/api/replay/files"
    ),
  getReplayFile: (filename: string) =>
    reqJson<TracePayload>(`/api/replay/files/${encodeURIComponent(filename)}`),
  getBenchmark: () => reqJson<BenchmarkReport>("/api/benchmark"),
  authMe: () => reqJson<AuthMeResponse>("/api/auth/me"),
  loginWithGoogle: (idToken: string, mode: "login" | "signup" = "login") =>
    reqJson<{
      authenticated: boolean;
      requires_email_verification?: boolean;
      email?: string;
      message?: string;
      user?: AuthUser;
      csrf_token?: string;
    }>("/api/auth/google", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id_token: idToken, mode }),
    }),
  requestEmailCode: (email: string, mode: "login" | "signup" = "login") =>
    reqJson<{ challenge_sent: boolean; email: string; expires_in_seconds: number }>(
      "/api/auth/email/request",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, mode }),
      }
    ),
  verifyEmailCode: (email: string, code: string, mode: "login" | "signup" = "login") =>
    reqJson<{ authenticated: boolean; user: AuthUser; csrf_token: string }>(
      "/api/auth/email/verify",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, code, mode }),
      }
    ),
  logout: () =>
    reqJson<{ authenticated: boolean }>("/api/auth/logout", {
      method: "POST",
    }),
};
