# CrewForge Roadmap

**Challenge:** SEED Hackathon 2026 — Chain of Command (Team Git Commit)  
**Target Working End-to-End Multi-Agent Engine:** November 27, 2026  
**Final Submission Deadline:** December 1, 2026  
**Remaining Cloud API Budget Cap:** \$6.10 total (development cap currently \$1.00 cumulative, \$0.05/run)

## Phase 1 — Engineering Foundation (November 25, 2026)

- [x] Workspace, Python 3.12 (`uv`), Ruff, and Pytest setup
- [x] Typed Pydantic v2 contracts (`AgentSpec`, `TaskSpec`, `ExecutionPlan`, `Claim`, `Evidence`, `AgentMessage`, `ReviewVerdict`, `RunConfig`, `RunResult`, `UsageMetrics`, `ExecutionEvent`)
- [x] Async `ModelProvider` abstraction (`OpenAIProvider` via Responses API, `OllamaProvider` via `/api/chat`)
- [x] Persistent, concurrency-safe SQLite storage and `BudgetManager`
- [x] Single-agent `BaselineRunner`
- [x] Official MCP Python SDK server (`baseline`, `trace`) over `stdio` and localhost `streamable-http`
- [x] Offline automated test suite

## Phase 2 — End-to-End Multi-Agent Engine (November 26–27, 2026)

- [ ] **Lead Planner & Dynamic Specialist Factory:** Prompt-agnostic `ExecutionPlan` generation and specialist instantiation
- [ ] **DAG Dependency Scheduler & Message Bus:** Concurrent `asyncio` task execution and persisted `AgentMessage` routing
- [ ] **Versioned Claim Ledger:** SQLite-backed `Claim`, `Evidence`, and `ReviewVerdict` persistence with version history
- [ ] **Independent Reviewer Pool & Targeted Revisions:** Automated detection of unsupported claims, calculation errors, and contradictions with bounded revision cycles
- [ ] **Honesty Gate:** Final synthesis verification preventing unmarked unsupported claims
- [ ] **MCP `solve(prompt)` Tool:** Expose `solve(prompt)` alongside `baseline(prompt)` and `trace(run_id)`
- [ ] **End-to-End Offline & Local (`qwen3:4b`) Verification:** Working multi-agent engine verified by **November 27, 2026**

## Phase 3 — Evaluation, Hardening & Submission (November 28–30, 2026)

- [ ] **David vs Goliath Evaluation Suite:** Fair benchmark comparing small open-weight multi-agent network (`qwen3:4b`) vs single-agent baseline within strict \$6.10 total budget
- [ ] **Trace & Cost Audit:** Verify reproducibility, latency, token efficiency, and Honesty Gate precision
- [ ] **Final Documentation & Demo Polish:** Finalize submission artifacts ahead of **December 1, 2026**
