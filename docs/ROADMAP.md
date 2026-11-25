# CrewForge Roadmap

**Challenge:** SEED Hackathon 2026 — Chain of Command (Team Git Commit)  
**Target Working End-to-End Multi-Agent Engine:** November 27, 2026  
**Final Submission Deadline:** December 1, 2026  
**Remaining Cloud API Budget Balance:** ~\$6.07 of \$6.10 preserved for final demo (`\$0.0255` Phase 4 cloud dev expenditure inside `\$0.25` cap)

## Phase 1 — Engineering Foundation (Completed November 25, 2026)

- [x] Workspace, Python 3.12 (`uv`), Ruff, and Pytest setup
- [x] Typed Pydantic v2 contracts & provider abstractions (`OpenAIProvider`, `OllamaProvider`)
- [x] Persistent SQLite ledger, `BudgetManager`, `BaselineRunner`, and MCP (`baseline`, `trace`)

## Phase 2 — Dynamic Multi-Agent Engine (Completed November 25, 2026)

- [x] `ModelExecutor` atomic accounting, shared semaphore, schema-aware estimation, SQLite v2
- [x] `LeadPlanner`, dependency-aware `DAGScheduler`, `SpecialistWorker`, `MessageBus`, `solve(prompt)`

## Phase 3 — Verification Engine, Targeted Repair & Honesty Gate (Completed November 25, 2026)

- [x] Versioned claim categories/verification states, `DeterministicArithmeticTool`, `ReviewerPool`, `TargetedRepairEngine`, `ClaimDependencyGraph`, `HonestyGate`, and local `qwen3:4b` baselines

## Phase 4 — Release Hardening, Portability & Cloud Pilots (Completed November 25, 2026)

- [x] **Claim-Locked Release:** Eliminated lexical-overlap Gate bypasses; closed laundering across conditional, recommendation, assumption, missing-info, unresolved, and task-error fields
- [x] **Resolvable Arithmetic Lineage:** Exact variable-to-target resolution (`up_k`, `doc_k`, `clm_k@vN`, `tlx_k`, `const:`, `assumption:`) rejecting swaps, fabricated refs, unrelated spans, stale refs, and ungrounded inline constants
- [x] **Strict Reviewer Validation & Coverage Separation:** Unique `(claim, version, reviewer, dim)` verdicts, strict claim-evidence matching, fresh premise recheck, and separated `execution_completed`/`content_released_safely`/`requirement_coverage`
- [x] **External Retrieval Foundation & Provider Portability:** `LocalDocumentRetriever` + allowlisted SSRF-protected `AllowlistedHttpRetriever` (disabled by default); strict OpenAI schemas, `reasoning.effort`, `returned_model`, `openai_compatible` adapter
- [x] **Cloud Guard, Deterministic Fast Path & `gpt-6-luna` Pilots:** `CREWFORGE_ALLOW_PAID_RUNS=false` default, `.crewforge/cloud_dev.db`, fast-path synthesis, strong & `--compute-matched` baselines, MCP smoke test + 4 matched cloud pilots

## Phase 5 — Competitive Evaluation & Engine Freeze (Completed November 25, 2026)

- [x] **Focused Stabilization:** Honesty Gate category enforcement, premise-chain requirement on recommendations, assumption requirement on conditional conclusions, auxiliary field unverified number/echo detection, document chunking & multi-chunk provenance catalog fix.
- [x] **Unified Persistent Budget Authority:** Canonical `.crewforge/budget_authority.db` reconciling pilot expenditures ($0.025516 committed across 55 unique calls), enforcing $4.00 emergency protected reserve, and hard allocation caps (`phase5_same_model` max $0.75, `david_vs_goliath` max $1.10).
- [x] **Reproducible Blind Benchmark Harness:** `src/crewforge/evaluation/` contracts, programmatic objective checks (exact numbers, units, percentage points, logical restrictions, missing info), cryptographic A/B blinding, 100-point rubric with unsupported assertion penalty (-0 to -20), multi-format reporting (JSON, JSONL, CSV, Markdown).
- [x] **True Iterative Compute-Matched Baseline:** Multi-round self-critique loop (`BaselineCritiqueOutput`, up to 3 rounds) with deterministic arithmetic parity, stopping early when no defects remain.
- [x] **Small-Model Profile & Portable Demo Trace:** Concurrency 1, max 2 workers, max 3 tasks for Apple Silicon M1; sanitized export in `crewforge.core.demo_trace`.
- [x] **Engine Freeze & Offline Verification:** 50/50 tests passing (including live HTTP MCP localhost roundtrip), zero ruff errors, and zero paid API expenditure during freeze.

## Final Dashboard, Replay & Live Submission Prep (November 26–December 1, 2026)

- [ ] **Visual Dashboard & Trace Replay:** Built strictly on persisted SQLite traces without burning live API balance
- [ ] **Live Finale Rehearsal & Submission Package:** Utilizing the preserved \$4.00+ OpenAI balance prior to December 1, 2026
