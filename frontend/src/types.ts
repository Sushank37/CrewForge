export type AppView = "chat" | "mission-control" | "settings" | "mcp" | "characters";
export type ThemeMode = "dark" | "light";
export type RunModeType = "network" | "baseline";

export interface AuthUser {
  user_id: string;
  email: string | null;
  email_verified: boolean;
  google_linked: boolean;
  display_name: string | null;
  created_at: string;
  last_login_at: string;
}

export interface AuthMeResponse {
  authenticated: boolean;
  auth_required: boolean;
  role?: string;
  user_id?: string;
  user?: AuthUser | null;
  csrf_token?: string | null;
  google_enabled: boolean;
  google_client_id?: string | null;
  email_delivery_configured?: boolean;
}

export interface ChatSummary {
  chat_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface AttachmentMeta {
  name: string;
  size: number;
}

export interface ChatMessage {
  message_id: string;
  chat_id: string;
  role: "user" | "assistant";
  mode: RunModeType;
  content: string;
  run_id: string | null;
  status: string;
  attachments: AttachmentMeta[];
  error: string | null;
  created_at: string;
  updated_at: string;
}

export interface ChatDetail extends ChatSummary {
  messages: ChatMessage[];
}

export interface JudgeSettings {
  billing_mode: "free" | "byok";
  free_tier_verified: boolean;
  provider: "ollama" | "openai" | "openai_compatible";
  model: string;
  openai_base_url: string;
  ollama_base_url: string;
  openai_reasoning_effort: "none" | "low" | "medium" | "high";
  allow_paid_runs: boolean;
  max_output_tokens: number;
  max_workers: number;
  max_tasks: number;
  adaptive_team_capacity?: boolean;
  max_concurrency: number;
  max_model_calls_per_run: number;
  max_cost_per_run_usd: number;
  total_budget_ceiling_usd: number;
  custom_input_price_per_1m: number | null;
  custom_output_price_per_1m: number | null;
  api_key_configured: boolean;
  api_key_hint: string | null;
  session_db_path: string;
  session_budget_db_path: string;
}

export interface McpStatus {
  tools: string[];
  discovered_tools?: string[];
  http_endpoint: string;
  http_reachable: boolean;
  handshake_error?: string | null;
  session_id?: string;
  remote_https?: boolean;
  session_db_path: string;
  session_budget_db_path?: string;
  shared_http_config?: Record<string, unknown>;
  stdio_config: Record<string, unknown>;
  http_command: string;
}

export interface TraceEvent {
  event_id?: string;
  seq: number;
  event_type: string;
  category?: string;
  actor?: string;
  message?: string;
  payload?: Record<string, unknown>;
  created_at?: string;
  timestamp?: string;
}

export interface TraceAgent {
  agent_id: string;
  role: string;
  specialty?: string;
  instructions?: string;
  model_override?: string;
  model?: string;
}

export interface TraceTask {
  task_id: string;
  title?: string;
  description?: string;
  assigned_agent_id: string;
  depends_on?: string[];
  depends_on_task_ids?: string[];
  status: string;
}

export interface TraceClaim {
  claim_id: string;
  version: number;
  statement: string;
  category: string;
  status: string;
  author_agent_id: string;
  evidence_ids?: string[];
  depends_on_claim_ids?: string[];
  premise_refs?: string[];
  tool_execution_ids?: string[];
  requirement_refs?: string[];
  created_at?: string;
}

export interface TraceEvidence {
  evidence_id: string;
  source_type: string;
  source_ref?: string;
  excerpt: string;
  verified_in_input?: boolean;
  created_at?: string;
}

export interface TraceTool {
  tool_execution_id: string;
  tool_name: string;
  expression?: string;
  variables?: Record<string, unknown>;
  computed_result?: string | null;
  units?: string | null;
  success?: boolean;
  inputs_grounded?: boolean;
  matches_proposed?: boolean | null;
  provenance_details?: Record<string, { ref: string; grounded: boolean }>;
  error?: string | null;
  created_at?: string;
}

export interface TraceVerdict {
  verdict_id: string;
  claim_id: string;
  claim_version: number;
  reviewer_id: string;
  review_dimension?: string;
  decision?: string;
  verdict?: string;
  rationale?: string;
  required_correction?: string | null;
  created_at?: string;
}

export interface TracePeerMessage {
  message_id: string;
  sender_id?: string;
  sender_agent_id?: string;
  recipient_id?: string;
  recipient_agent_id?: string;
  message_type?: string;
  content: string;
  created_at?: string;
}

export interface TracePayload {
  run_id: string;
  mode: string;
  status: string;
  provider: string;
  model: string;
  returned_model?: string | null;
  is_historical_replay: boolean;
  latest_seq?: number;
  incremental?: boolean;
  waiting_for_provider?: boolean;
  waiting_reason?: string | null;
  provider_wait_seconds?: number | null;
  created_at?: string;
  completed_at?: string | null;
  latency_ms?: number | null;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    reasoning_tokens?: number;
    actual_cost_usd?: number;
    model_calls?: number;
    usage_reported?: boolean;
  };
  roster: TraceAgent[];
  tasks: TraceTask[];
  task_transitions?: Array<{ task_id: string; to_status: string; created_at: string }>;
  execution_events: TraceEvent[];
  peer_messages: TracePeerMessage[];
  evidence: TraceEvidence[];
  claims: TraceClaim[];
  review_verdicts: TraceVerdict[];
  tool_executions: TraceTool[];
  honesty_gate?: {
    passed: boolean;
    release_mode: string;
    reasons?: string[];
    rejected_segments?: unknown[];
    rejected_claim_refs?: unknown[];
    verification_summary?: {
      supported_claim_count?: number;
      rejected_claim_count?: number;
      requirements_unresolved?: number;
      requirements_partial?: number;
      requirement_coverage?: number;
      execution_completed?: boolean;
      content_released_safely?: boolean;
      requirement_statuses?: Array<{
        requirement_id: string;
        text: string;
        status: string;
        rationale: string;
      }>;
    };
  } | null;
  final_result?: {
    answer?: string;
    missing_information?: string[];
    verification?: {
      requirement_coverage?: number;
    };
  } | null;
}

export interface BenchmarkReport {
  grading_method: string;
  benchmark_note: string;
  engine_commit: string;
  model: string;
  summary: {
    baseline_mean: number;
    baseline_median: number;
    network_mean: number;
    network_median: number;
    network_wins: number;
    network_losses: number;
    ties: number;
    baseline_unsupported_assertions: number;
    network_unsupported_assertions: number;
    baseline_total_cost_usd: number;
    network_total_cost_usd: number;
    baseline_mean_latency_s: number;
    network_mean_latency_s: number;
  };
  domain_scores: Record<string, { baseline_mean: number; network_mean: number }>;
  per_case: Array<{
    prompt_id: string;
    domain: string;
    winner: string;
    baseline: { final_score: number; unsupported_material_assertion_count: number };
    network: { final_score: number; status: string; unsupported_material_assertion_count: number };
  }>;
  david_vs_goliath: { status: string; detail?: string };
}
