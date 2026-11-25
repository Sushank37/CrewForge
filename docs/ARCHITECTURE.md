# CrewForge Architecture

## 1. Phase 1 Foundation

Phase 1 establishes the deterministic infrastructure beneath the multi-agent engine:

```text
MCP Client (stdio / localhost streamable-http)
       │
       ▼
crewforge.mcp_server (FastMCP: baseline, trace)
       │
       ├──► BaselineRunner ──► BudgetManager ──► SQLite (runs, events, usage, reservations)
       │         │
       │         ▼
       │    ModelProvider (OpenAI Responses API | Ollama /api/chat)
       │
       └──► Database.get_run_trace(run_id)
```

### Key Design Properties

- **Model-Agnostic Interface:** `ModelProvider` standardizes text and Pydantic-validated structured JSON generation across cloud (`OpenAIProvider`) and local (`OllamaProvider`) inference without hard-coded model names or domain-specific routing.
- **Fail-Closed Budget Governance:** `BudgetManager` wraps SQLite `BEGIN IMMEDIATE` transactions so concurrent async tasks cannot race past `max_model_calls_per_run`, `max_cost_per_run_usd`, or `max_cumulative_cost_usd`. Paid models without explicit pricing fail closed prior to any network call. Network timeouts or mid-flight errors with unknown billing outcomes conservatively retain their reserved cost.
- **Transport Isolation:** Storage, budgeting, and execution logic have zero dependency on MCP transport layers. MCP operational logging is pinned to `stderr` so `stdio` protocol framing stays clean.

## 2. Target Multi-Agent Architecture (Phase 2)

Phase 2 builds `solve(prompt)` on top of the Phase 1 contracts (`ExecutionPlan`, `AgentSpec`, `TaskSpec`, `Claim`, `Evidence`, `AgentMessage`, `ReviewVerdict`):

1. **Dynamic Lead Planner:** Analyzes an arbitrary prompt and produces an `ExecutionPlan` defining bespoke `AgentSpec` specialists and a DAG of `TaskSpec` nodes with explicit `depends_on` edges. No topic-specific roles or routing rules are hard-coded.
2. **Dependency Scheduler:** Executes ready tasks concurrently up to `CREWFORGE_MAX_CONCURRENCY` using `asyncio` semaphores, passing completed upstream outputs and claims to downstream dependents.
3. **Shared Message Bus:** Delivers `AgentMessage` records (direct or broadcast) across active specialists and persists them in the run trace for full replayability.
4. **Versioned Claim Ledger:** Specialists emit `Claim` objects linked to `Evidence` provenance and upstream `depends_on_claim_ids`. Every claim starts as `asserted` and increments `version` (linking `parent_claim_id`) upon revision.
5. **Independent Reviewer Pool:** Reviewers inspect claims, calculations, evidence citations, and cross-claim consistency, emitting `ReviewVerdict` records (`approve`, `reject`, `needs_revision`, `unresolved`).
6. **Targeted Revision Loop:** Rejected or needs-revision claims are routed back to their originating specialists with reviewer `blocking_issues` up to a bounded revision depth under the shared `BudgetManager`.
7. **Honesty Gate:** Before returning the final `RunResult`, the Honesty Gate audits all claims referenced in the final synthesis:
   - `supported` claims pass directly.
   - `rejected` claims are excluded or explicitly flagged as refuted.
   - `asserted` / `unresolved` claims cannot be presented as verified fact and must be marked as unverified or surfaced in `missing_information` / `confidence_notes`.
8. **David vs Goliath Evaluation:** Because `solve(prompt)` and `baseline(prompt)` share identical provider, timeout, token, and tool boundaries, a small open-weight networked configuration (e.g., `qwen3:4b`) can be benchmarked fairly against a stronger single-agent baseline.
