# CrewForge Architecture

## 1. Phase 3 Runtime Architecture

CrewForge exposes three MCP tools (`solve`, `baseline`, `trace`) and matching CLI commands backed by a transport-independent core engine and SQLite persistence layer. In MCP server mode, `solve` and `baseline` share one `ModelExecutor` instance so concurrent requests obey global concurrency and budget constraints:

```text
MCP Client (stdio / localhost streamable-http)  or  CLI
       │
       ├──► solve(prompt) ──► NetworkRunner (shared ModelExecutor)
       │                         ├── LeadPlanner (ExecutionPlan + <=1 repair)
       │                         ├── DAGScheduler (asyncio DAG execution)
       │                         ├── SpecialistWorker + MessageBus + DeterministicArithmeticTool + ClaimLedger
       │                         ├── Bounded Follow-up Round (<= max_communication_rounds)
       │                         ├── ReviewerPool (deterministic prechecks + independent review)
       │                         ├── TargetedRepairEngine + ClaimDependencyGraph (v+1 & stale cascade)
       │                         └── LeadSynthesizer + HonestyGate (enforced on full & partial paths)
       │
       ├──► baseline(prompt) ──► BaselineRunner (with DeterministicArithmeticTool parity) ──► ModelExecutor
       │
       └──► trace(run_id) ──► Database.get_run_trace(run_id)  [no model provider needed]
```

## 2. `ModelExecutor` & Per-Attempt Budget Accounting

All baseline, planner, worker, follow-up, reviewer, repair, and synthesis invocations pass through `ModelExecutor`:

- **Shared Concurrency Semaphore & Deadlines:** One `asyncio.Semaphore(settings.effective_max_concurrency)` is shared across the executor (`1` by default for local Ollama on 8 GB RAM; `3` for cloud). Semaphore acquisition is bounded by the remaining `run_deadline_monotonic` so queued tasks fail cleanly on timeout rather than hanging.
- **Per-Attempt Atomic Reservations:** Each individual model attempt reserves budget in SQLite (`BEGIN IMMEDIATE`) *before* dispatch, estimating input tokens across system prompt, user/task prompt, and serialized Pydantic JSON response schema.
- **Retry Ownership:** Provider/SDK retries are forced to `0` when called by `ModelExecutor`; any retry configured via `CREWFORGE_MAX_RETRIES` is orchestrated by `ModelExecutor` with its own fresh reservation.
- **Exact-Once Reconciliation & Accounting:**
  - Completed calls reconcile actual usage; if a paid provider omits usage (`usage_reported=False`), `BudgetManager` retains the full reserved cost (`status='unknown'`, `outcome='completed_unknown_usage'`) rather than recording `\$0.00`.
  - Dispatched calls that fail with known-unbilled outcomes (`status='released'`) still count toward `max_model_calls_per_run`.
  - Timeouts, cancellations, or ambiguous failures retain full reserved cost (`status='unknown'`).

## 3. Dynamic Planning, Compact Local Limits & DAG Scheduling

1. **`LeadPlanner`** produces a compact `PlannerOutput` using short local handles (`agent_ref`, `task_ref`), validated for uniqueness, valid assignments, valid dependency references, DAG acyclicity (Kahn's algorithm), capability allowlists, no `model_override`, and bounded counts (`effective_max_workers=3` and `effective_max_tasks=3` for local Ollama; `4`/`6` otherwise). At most one budgeted repair call is permitted; invalid plans never fall back to a hard-coded domain team.
2. **`DAGScheduler`** executes independent tasks concurrently via `asyncio.create_task` and `asyncio.wait(..., return_when=FIRST_COMPLETED)`, reserving call headroom for review/synthesis and logging `optional_stage_skipped` when optional stages cannot fit inside budget or remaining deadline. Failed tasks immediately transition transitive dependents to `blocked` while safe independent branches finish. Total `run_deadline_seconds` cancels overrunning tasks cleanly and triggers an explicit `partial` run status.

## 4. Peer Messaging & ID/Version-Targeted Claim Revisions

`MessageBus` persists `agent_messages` and per-recipient `message_deliveries` rows (`pending -> delivered -> consumed`) within the run:

- Messages to specialists, reviewers, and the Lead (`'lead'`) are validated against current-run actors and bounded by `max_message_chars`.
- `LeadSynthesizer` explicitly reads and injects Lead inbox messages into synthesis context and marks them `consumed` only after the model call executes.
- Follow-ups (`FollowUpOutput`) and targeted repairs (`RevisionOutput`) must address claims by explicit `(target_claim_id, expected_version)` with same-author and same-run enforcement — never by list position — preserving immutable prior claim versions (`v1`, `v2`, ...) in SQLite.

## 5. Claim Taxonomy, Deterministic Tools, Review & Honesty Gate

- **Claim & Verification Separation:** Claims carry `ClaimCategory` (`given_fact`, `external_factual_assertion`, `computation`, `logical_inference`, `assumption`, `recommendation`, `uncertainty`), runtime `ClaimStatus` (`asserted`, `supported`, `rejected`, `unresolved`, `stale`), separate `VerificationStatus` (`provenance_verified`, `computationally_verified`, `independently_reviewed_inference`, `unsupported`, `contradicted`, `stale_pending_review`), and optional `confidence` (`0.0..1.0`).
- **`DeterministicArithmeticTool` & Baseline Parity:** Evaluates `Decimal` (precision 28) expressions over a strict allowlisted AST (no `eval`/`exec`, bounded depth/nodes/magnitude/exponents), strips common numeric formatting (`$`, `₹`, `€`, `£`, `¥`, commas), checks numeric input grounding against user/dependency text, and records immutable `tool_executions` rows. `BaselineRunner` exposes matching deterministic `ArithmeticRequest` execution for fair comparison. Passing evaluation + input grounding still requires reviewer approval that the formula and units match the user's question.
- **Evidence Provenance (`evidence_verifier.py`):** Validates `user_prompt` and `dependency_finding` excerpts for normalized substring match, numeric preservation, and surrounding negation omission; `UnavailableExternalRetriever` keeps external citations `unverified` (`external_unverified`) with no hidden internet fetches.
- **Independent `ReviewerPool` & `ClaimDependencyGraph`:** Deterministic prechecks run first before budgeted reviewer model calls (`reviewer_combined` in compact mode, or separate `reviewer_evidence` / `reviewer_consistency`). Both code and SQLite forbid `reviewer_id == author_agent_id`, missing verdicts keep material claims `unresolved`, and `ClaimDependencyGraph` rejects self/cross-run/cyclic `premise_refs` while transitively invalidating downstream claims (`stale` / `stale_pending_review`) on upstream rejection or revision.
- **Deterministic `HonestyGate`:** Audited across full and partial release paths, rejecting empty/unknown/cross-run/stale/rejected/ungrounded segment citations, unapproved `computation` or unretrieved `external_factual_assertion` claims, unverified numbers/factual text emitted by synthesis, and ungrounded `new_factual_assertions`, persisting `honesty_gate_decisions` and returning `VerificationSummary`.
- **SQLite Schema Migrations:** Repeatable migrations track `v1` -> `v2` -> `v3`, adding `run_reviewers`, `tool_executions`, `honesty_gate_decisions`, and v3 columns on `claims` / `review_verdicts` without losing historical runs.

## 6. Known Limitations

- Automated verification constrains fabricated numbers, self-reviews, and unreviewed releases, but semantic or conceptual errors in natural-language inferences can still pass if independent reviewers erroneously approve them.
- `UnavailableExternalRetriever` leaves external empirical citations `unverified` until an authorized retrieval backend is integrated.
- On 8 GB Apple Silicon running local `qwen3:4b`, multi-call verification runs typically take 5–7 minutes (`CREWFORGE_RUN_DEADLINE_SECONDS=480.0`) and may produce `partial` releases if a specialist hits token truncation or a claim fails post-repair re-review.
