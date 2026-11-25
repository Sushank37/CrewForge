# CrewForge Roadmap

**Challenge:** SEED Hackathon 2026 — Chain of Command (Team Git Commit)  
**Target Working End-to-End Multi-Agent Engine:** November 27, 2026  
**Final Submission Deadline:** December 1, 2026  
**Remaining Cloud API Budget Cap:** \$6.10 total (zero paid expenditure incurred through Phase 2)

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

## Phase 3 — Verification, Honesty Gate & Evaluation (November 26–30, 2026)

- [ ] **Independent Reviewer Pool:** Evaluate claims for unsupported assertions, arithmetic/logic errors, and cross-claim contradictions (`ReviewVerdict`)
- [ ] **Deterministic Verification Tools:** Arithmetic/expression checker and verifiable evidence matching
- [ ] **Targeted Revision Loop:** Route `rejected` / `needs_revision` claims back to originating specialists for bounded versioned repair
- [ ] **Honesty Gate:** Block unmarked unsupported claims from release
- [ ] **David vs Goliath Evaluation Suite:** Fair comparison of `qwen3:4b` multi-agent network against stronger single-agent baselines within the \$6.10 cloud budget prior to December 1, 2026
