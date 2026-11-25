# CrewForge Architecture

## 1. Phase 4 Runtime Architecture

CrewForge exposes three MCP tools (`solve`, `baseline`, `trace`) and matching CLI commands backed by a transport-independent core engine and SQLite persistence layer. In MCP server mode, `solve` and `baseline` share one `ModelExecutor` instance so concurrent requests obey global concurrency and budget constraints:

```text
MCP Client (stdio / localhost streamable-http)  or  CLI
       │
       ├──► solve(prompt, docs) ──► NetworkRunner (shared ModelExecutor + CREWFORGE_ALLOW_PAID_RUNS)
       │                         ├── LocalDocumentRetriever (shared operator corpus ingestion)
       │                         ├── LeadPlanner (ExecutionPlan + <=1 repair, req_1..req_K IDs)
       │                         ├── DAGScheduler (asyncio DAG execution)
       │                         ├── SpecialistWorker + MessageBus + DeterministicArithmeticTool + ClaimLedger
       │                         ├── Selective Follow-up Round (skipped immediately if zero unread msgs)
       │                         ├── ReviewerPool (strict unique verdicts + alias resolution + premise recheck)
       │                         ├── TargetedRepairEngine + ClaimDependencyGraph (v+1 & stale cascade)
       │                         └── LeadSynthesizer (deterministic fast path or synthesis) + Claim-Locked HonestyGate
       │
       ├──► baseline(prompt, docs) ──► BaselineRunner (tool parity + optional --compute-matched) ──► ModelExecutor
       │
       └──► trace(run_id) ──► Database.get_run_trace(run_id)  [no model provider needed]
```

## 2. `ModelExecutor`, Cloud Guard & Per-Attempt Accounting

All baseline, planner, worker, follow-up, reviewer, repair, and synthesis invocations pass through `ModelExecutor`:

- **Paid-Run Guard & Cloud DB Isolation:** `CREWFORGE_ALLOW_PAID_RUNS` defaults to `false`; `BudgetManager.reserve_call` rejects paid provider calls before reservation unless explicitly enabled, fails closed without matching pricing in `CREWFORGE_PRICING_JSON`, and routes paid runs cleanly to `cloud_db_path` (`.crewforge/cloud_dev.db`) when using default local config.
- **Portable Adapters & Parameter Gating:** `OpenAIProvider` supports Responses API strict JSON schemas (`ArithmeticVariableBinding` arrays instead of dynamic dictionaries) with `reasoning={"effort": ...}`, omits unsupported `temperature` on reasoning models (`gpt-6-luna`), captures actual `returned_model`, and supports explicit `openai_compatible` Chat Completions servers via operator-controlled `openai_base_url`. `OllamaProvider` enforces `num_ctx`, `think`, and captures returned model names.
- **Shared Concurrency Semaphore & Deadlines:** One `asyncio.Semaphore(settings.effective_max_concurrency)` (`1` for Ollama, `3` for OpenAI) is shared across calls with deadline-bounded acquisition. Every attempt reserves input + schema + output budget atomically, counts unbilled released calls, and preserves reserved cost on unknown outcomes.

## 3. Dynamic Planning, Compact Local Limits & DAG Scheduling

1. **`LeadPlanner`** produces a compact `PlannerOutput` using short local handles (`agent_ref`, `task_ref`), validated for uniqueness, valid assignments, valid dependency references, DAG acyclicity (Kahn's algorithm), capability allowlists, no `model_override`, and bounded counts (`effective_max_workers=3` and `effective_max_tasks=3` for local Ollama; `4`/`6` otherwise). At most one budgeted repair call is permitted; invalid plans never fall back to a hard-coded domain team.
2. **`DAGScheduler`** executes independent tasks concurrently via `asyncio.create_task` and `asyncio.wait(..., return_when=FIRST_COMPLETED)`, reserving call headroom for review/synthesis and logging `optional_stage_skipped` when optional stages cannot fit inside budget or remaining deadline. Failed tasks immediately transition transitive dependents to `blocked` while safe independent branches finish. Total `run_deadline_seconds` cancels overrunning tasks cleanly and triggers an explicit `partial` run status.

## 4. Peer Messaging & ID/Version-Targeted Claim Revisions

`MessageBus` persists `agent_messages` and per-recipient `message_deliveries` rows (`pending -> delivered -> consumed`) within the run:

- Messages to specialists, reviewers, and the Lead (`'lead'`) are validated against current-run actors and bounded by `max_message_chars`.
- `LeadSynthesizer` explicitly reads and injects Lead inbox messages into synthesis context and marks them `consumed` only after the model call executes.
- Follow-ups (`FollowUpOutput`) and targeted repairs (`RevisionOutput`) must address claims by explicit `(target_claim_id, expected_version)` with same-author and same-run enforcement — never by list position — preserving immutable prior claim versions (`v1`, `v2`, ...) in SQLite.

## 5. Claim-Locked Gate, Exact Lineage, Reviewer Deduplication & Retrieval

- **Claim & Verification Separation:** Claims carry `ClaimCategory` (`given_fact`, `external_factual_assertion`, `computation`, `logical_inference`, `assumption`, `recommendation`, `uncertainty`), runtime `ClaimStatus`, separate `VerificationStatus`, and optional `confidence` (`0.0..1.0`).
- **Exact Arithmetic Provenance Lineage (`DeterministicArithmeticTool`):** Evaluates bounded `Decimal` expressions over a strict AST allowlist and resolves every variable via `ProvenanceCatalog` to `up_k` prompt spans, `doc_k`/`evd_...` verified excerpts, current `SUPPORTED` claim versions, prior valid `tlx_...` records, or declared `const:`/`assumption:` markers — detecting variable-to-number swaps, fabricated targets, unrelated sentence matches, ungrounded AST literals, unit conflicts, and stale premise versions. `BaselineRunner` receives matching tools, documents, and optional single-agent `--compute-matched` self-check refinement.
- **Evidence & External Retrieval (`retrieval.py`, `evidence_verifier.py`):** `LocalDocumentRetriever` ingests operator-supplied files/text into immutable `Evidence` with SHA-256 hashes, timestamps, and character offsets. `AllowlistedHttpRetriever` stays disabled by default and blocks non-HTTPS URLs, private/SSRF DNS resolutions, redirects, and credential forwarding. No open-web search engine exists in Phase 4.
- **Strict Reviewer Deduplication (`ReviewerPool`):** Allows at most one verdict per `(claim_id, claim_version, reviewer_id, review_dimension)` per round, validates cited `evidence_refs` against the claim's own evidence, treats skipped/missing dimensions and conflicting verdicts as `UNRESOLVED`/`REJECTED`, and rechecks live premise currency before promoting any claim to `SUPPORTED`.
- **Claim-Locked `HonestyGate` & Coverage Separation:** Renders approved facts verbatim from exact ledger claim versions, blocks paraphrases/negation reversals and laundering across all segment and auxiliary fields, and reports `requirement_coverage` via explicit `req_k` linkage alongside separate `execution_completed`, `release_checks_executed`, `content_released_safely`, and `independent_grade` fields. When requirements are fully covered without urgent Lead inbox items, `LeadSynthesizer` can execute the deterministic synthesis fast path (`synthesis_mode="deterministic_fast_path"`), still audited by the same `HonestyGate`.
- **SQLite Schema Migrations (`v1` -> `v4`):** Schema v4 idempotently adds retrieval metadata (`content_hash`, `retrieved_at`, `source_location`, `retrieval_status`) and tool lineage columns (`provenance_details_json`, `has_ungrounded_literals`, `formula_appropriate`).

## 6. Known Limitations

- Automated verification constrains fabricated numbers, self-reviews, and claim rewrites, but subtle conceptual errors inside natural-language claims approved by reviewers can still pass.
- No open-web search exists; external assertions without supplied documents remain `unverified` (`external_unverified`) when HTTP retrieval is disabled or blocked.
- Internal `requirement_coverage` reflects claim-to-requirement linkage within CrewForge and is explicitly separated from independent correctness grading (`independent_grade=None`).
- Transport-layer continuous IP pinning is not implemented; pre-connect SSRF checks protect against intranet exfiltration.

## 7. Phase 5 Unified Budget Authority & Epistemic Enforcement

- **Canonical Budget Authority (`.crewforge/budget_authority.db`):** Reconciles past reservations from `.crewforge/crewforge.db` and `.crewforge/cloud_dev.db` by unique `reservation_id` (preventing double counting). Enforces the hackathon-mandated $4.00 emergency protected reserve (from $6.10 balance) and hard category caps (`phase5_same_model` max $0.75, `david_vs_goliath` max $1.10).
- **Epistemic Category Enforcement:** Honesty Gate prevents unsupported factual claims from being released under `recommendations` or `conditional_conclusions`. Recommendations require verified/supported material premises. Conditional conclusions must explicitly declare assumptions. Auxiliary fields reject unverified numbers and echoes of rejected claims.
- **Robust Document Chunking & Provenance:** Documents are chunked without mid-word slicing, and multi-chunk documents are aliased cleanly to `doc_1..doc_N` so Arithmetic provenance checks ground expressions accurately across document bodies.

## 8. Blind Benchmark Evaluation & Safe Trace Export

- **Cryptographic A/B Blinding:** Evaluators review blind packets with internal claim/evidence tags stripped, ensuring zero judge bias between single-agent and multi-agent outputs.
- **100-Point Scoring & Penalty:** Programmatic objective checks (exact numbers, percentages vs percentage points, logical restrictions, missing info) combined with a 100-point rubric and an explicit negative penalty (-0 to -20) for unsupported assertions.
- **True Iterative Baseline:** `BaselineRunner` implements multi-round self-critique (`BaselineCritiqueOutput`, up to 3 rounds) with deterministic arithmetic parity, stopping early when no defects remain.
- **Safe Trace Replay:** `export_demo_trace` produces sanitized JSON execution logs with API keys and system paths stripped.
