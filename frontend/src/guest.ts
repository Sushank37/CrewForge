import type {
  RunModeType, TraceAgent, TraceEvent, TraceEvidence, TracePayload, TracePeerMessage,
  TraceTask, TraceTool, TraceVerdict,
} from "./types";

interface GuestStatus {
  task_id: string;
  mode: RunModeType;
  status: string;
  current_stage: string;
  provider: string;
  model: string;
  created_at: string;
  completed_at: string | null;
  error?: string | null;
}

interface GuestTrace {
  trace_available?: boolean;
  agents: TraceAgent[];
  assignments: TraceTask[];
  messages: TracePeerMessage[];
  events: TraceEvent[];
  tool_executions: TraceTool[];
  review_decisions: TraceVerdict[];
  honesty_gate: {
    passed: boolean;
    release_mode: string;
    execution_completed?: boolean;
    content_released_safely?: boolean;
  } | null;
}

interface Source {
  source_ref: string;
  source_type: string;
  title?: string;
}

interface GuestResult {
  answer: string;
  conversational_answer?: string;
  returned_model?: string | null;
  sources: Source[];
  evidence: TraceEvidence[];
  verification: {
    independent_verification_performed: boolean;
    honesty_gate_passed?: boolean | null;
    release_mode: string;
    answer_complete?: boolean | null;
    content_released_safely?: boolean;
  };
  unresolved_requirements: Array<{ requirement_id: string; text: string; rationale?: string }>;
  usage?: {
    total_tokens: number;
    actual_cost_usd?: number | null;
    model_calls: number;
    usage_reported: boolean;
  };
}

export interface GuestJob {
  status: GuestStatus;
  trace: GuestTrace | null;
  result: GuestResult | null;
}

export function isActive(job: GuestJob | null): boolean {
  return job?.status.status === "queued" || job?.status.status === "running";
}

export function officeTrace(job: GuestJob | null): TracePayload | null {
  if (!job?.trace || job.trace.trace_available === false) return null;
  const { status, trace } = job;
  return {
    run_id: status.task_id,
    mode: status.mode,
    status: status.status,
    provider: status.provider,
    model: status.model,
    is_historical_replay: false,
    created_at: status.created_at,
    completed_at: status.completed_at,
    usage: job.result?.usage && {
      ...job.result.usage, actual_cost_usd: job.result.usage.actual_cost_usd ?? undefined,
    },
    roster: trace.agents,
    tasks: trace.assignments,
    execution_events: trace.events,
    peer_messages: trace.messages,
    review_verdicts: trace.review_decisions,
    tool_executions: trace.tool_executions,
    claims: [],
    evidence: job.result?.evidence ?? [],
    honesty_gate: trace.honesty_gate && {
      ...trace.honesty_gate,
      verification_summary: {
        execution_completed: trace.honesty_gate.execution_completed,
        content_released_safely: trace.honesty_gate.content_released_safely,
      },
    },
  };
}

export class GuestRequestError extends Error {
  constructor(message: string, readonly status: number) { super(message); }
}

export async function guestRequest<T>(
  path: string, options?: RequestInit,
): Promise<T> {
  const response = await fetch(`/api/guest${path}`, { cache: "no-store", ...options });
  if (response.ok) return response.json() as Promise<T>;
  const data = await response.json().catch(() => null);
  throw new GuestRequestError(
    data?.error || data?.message || "The guest service could not complete this request.",
    response.status,
  );
}

export function guestPost<T>(path: string, data: Record<string, unknown>): Promise<T> {
  return guestRequest<T>(path, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data),
  });
}
