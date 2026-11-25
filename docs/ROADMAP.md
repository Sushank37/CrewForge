# CrewForge Roadmap

**Challenge:** SEED Hackathon 2026 — Chain of Command (Team Git Commit)  
**Target Working End-to-End Multi-Agent Engine:** November 27, 2026  
**Final Submission Deadline:** December 1, 2026  
**Remaining Cloud API Budget Cap:** \$6.10 total (zero paid expenditure incurred through Phase 3)

## Phase 1 — Engineering Foundation (Completed November 25, 2026)

- [x] Workspace, Python 3.12 (`uv`), Ruff, and Pytest setup
- [x] Typed Pydantic v2 contracts (`AgentSpec`, `TaskSpec`, `ExecutionPlan`, `Claim`, `Evidence`, `AgentMessage`, `ReviewVerdict`, `RunConfig`, `RunResult`, `UsageMetrics`, `ExecutionEvent`)
- [x] Async `ModelProvider` abstraction (`OpenAIProvider` via Responses API, `OllamaProvider` via `/api/chat`)
- [x] Persistent SQLite storage and `BudgetManager`
- [x] Single-agent `BaselineRunner` and MCP server (`baseline`, `trace`)

## Phase 2 — Dynamic Multi-Agent Engine (Completed November 25, 2026)

- [x] **`ModelExecutor` & Accounting Fixes:** Per-attempt atomic reservations, shared concurrency semaphore, released-call counting, schema-aware input estimation, conservative missing-usage handling, error/secret sanitization
- [x] **SQLite Migrations & Ledger Persistence:** Schema v2 tables (`execution_plans`, `agents`, `tasks`, `task_transitions`, `evidence`, `claims`, `agent_messages`, `message_deliveries`, `review_verdicts`) with cross-run isolation and immutable claim versioning
- [x] **Dynamic Lead Planner:** Domain-agnostic task & specialist generation, structural & cycle validation, single budgeted repair attempt
- [x] **Dependency-Aware DAG Scheduler:** Concurrent `asyncio` execution, dependency failure propagation (`blocked`), run deadline enforcement, synthesis call reservation
- [x] **Specialist Workers, `MessageBus`, & `LeadSynthesizer`:** Bounded task/follow-up contexts, peer message delivery & consumption tracking, provenance-checked evidence proposals, validated synthesis references, `solve(prompt)` MCP & CLI integration
- [x] **Local `qwen3:4b` Smoke & Network Verification:** Offline suite + local Ollama execution verified ahead of November 27 target

## Phase 3 — Verification Engine, Targeted Repair & Honesty Gate (Completed November 25, 2026)

- [x] **Phase 2 Defect Fixes:** ID/version-targeted follow-ups (replacing positional revision), Lead inbox consumption, versioned `premise_refs` dependency graph, and strict gate controls on both full and partial synthesis paths
- [x] **Claim Contracts & SQLite Schema v3:** Added `ClaimCategory`, `VerificationStatus`, `ToolExecutionRecord`, `HonestyGateRecord`, `VerificationSummary`, `run_reviewers`, `tool_executions`, and `honesty_gate_decisions`
- [x] **Deterministic Verification & Fair Baseline Parity:** Safe `Decimal` AST `DeterministicArithmeticTool` (available to both network specialists and `BaselineRunner`) and deterministic provenance/negation checker + `EvidenceRetriever` interface
- [x] **Independent `ReviewerPool` & `TargetedRepairEngine`:** Author/reviewer independence enforcement, missing-verdict conservative `UNRESOLVED` handling, blocking objection routing over `MessageBus`, immutable `v+1` revisions, and transitive `ClaimDependencyGraph` stale invalidation
- [x] **Enforceable `HonestyGate` & Compact Local Profile:** Verified release filtering, live localhost Streamable HTTP MCP roundtrip test, offline adversarial test suite, and local `qwen3:4b` evaluations (`Test A` payback/attractiveness vs baseline, `Test B` incident response, `Test C` pass-rate causation)

## Final Evaluation & Submission Readiness (November 26–December 1, 2026)

- [ ] **David vs Goliath Cloud Comparison:** Authorized final benchmark comparing local `qwen3:4b` network runs against stronger single-agent cloud baselines within the untouched \$6.10 API cap
- [ ] **Optional Retrieval Backend & Packaging Polish:** Optional external `EvidenceRetriever` integration and final submission package prior to December 1, 2026
