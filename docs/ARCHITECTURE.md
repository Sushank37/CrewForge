# CrewForge Architecture

## 1. Phase 2 Runtime Architecture

CrewForge exposes three MCP tools (`solve`, `baseline`, `trace`) and matching CLI commands backed by a transport-independent core engine and SQLite persistence layer:

```text
MCP Client (stdio / localhost streamable-http)  or  CLI
       │
       ├──► solve(prompt) ──► NetworkRunner
       │                         ├── LeadPlanner (ExecutionPlan + <=1 repair)
       │                         ├── DAGScheduler (asyncio + shared ModelExecutor)
       │                         ├── SpecialistWorker + MessageBus + ClaimLedger
       │                         ├── Bounded Follow-up Round (<= max_communication_rounds)
       │                         └── LeadSynthesizer (validated claim refs / partial fallback)
       │
       ├──► baseline(prompt) ──► BaselineRunner ──► ModelExecutor
       │
       └──► trace(run_id) ──► Database.get_run_trace(run_id)  [no model provider needed]
```

## 2. `ModelExecutor` & Per-Attempt Budget Accounting

All baseline, planner, worker, follow-up, and synthesis invocations pass through `ModelExecutor`:

- **Shared Concurrency Semaphore:** One `asyncio.Semaphore(settings.effective_max_concurrency)` is shared across the executor so concurrent DAG branches never exceed `max_concurrency` (`1` by default for local Ollama on 8 GB RAM; `3` for cloud).
- **Per-Attempt Atomic Reservations:** Each individual model attempt reserves budget in SQLite (`BEGIN IMMEDIATE`) *before* dispatch, estimating input tokens across system prompt, user/task prompt, and serialized Pydantic JSON response schema.
- **Retry Ownership:** Provider/SDK retries are forced to `0` when called by `ModelExecutor`; any retry configured via `CREWFORGE_MAX_RETRIES` is orchestrated by `ModelExecutor` with its own fresh reservation.
- **Exact-Once Reconciliation & Accounting:**
  - Completed calls reconcile actual usage; if a paid provider omits usage (`usage_reported=False`), `BudgetManager` retains the full reserved cost (`status='unknown'`, `outcome='completed_unknown_usage'`) rather than recording `$0.00`.
  - Dispatched calls that fail with known-unbilled outcomes (`status='released'`) still count toward `max_model_calls_per_run`.
  - Timeouts, cancellations, or ambiguous failures retain full reserved cost (`status='unknown'`).

## 3. Dynamic Planning & DAG Scheduling

1. **`LeadPlanner`** produces a compact `PlannerOutput` using short local handles (`agent_ref`, `task_ref`), validated for uniqueness, valid assignments, valid dependency references, DAG acyclicity (Kahn's algorithm), capability allowlists, no `model_override`, and bounded counts (`max_workers`, `max_tasks`). At most one budgeted repair call is permitted; invalid plans never fall back to a hard-coded domain team.
2. **`DAGScheduler`** executes independent tasks concurrently via `asyncio.create_task` and `asyncio.wait(..., return_when=FIRST_COMPLETED)`, reserving call headroom (`reserve_synthesis_calls=1`) for synthesis. Failed tasks immediately transition transitive dependents to `blocked` while safe independent branches finish. Total `run_deadline_seconds` cancels overrunning tasks cleanly and triggers an explicit `partial` run status.

## 4. Peer Messaging Delivery Semantics

`MessageBus` persists `agent_messages` and per-recipient `message_deliveries` rows (`pending -> delivered -> consumed`) within the run:

- Specialist outgoing messages are validated against current-run recipients (specialists, `'lead'`, or broadcast `None`) and bounded by `max_message_chars` and per-turn count.
- Dependent tasks receive pending messages in their initial task prompt before execution.
- After DAG execution, up to `max_communication_rounds` (default `1`) executes bounded follow-up calls for specialists with unread peer messages, incorporating message content into the recipient's prompt and marking messages `consumed` only after model invocation.

## 5. Claim, Evidence, and SQLite Migration Design

- **Repeatable Migrations:** `schema_migrations` tracks schema versions (`v1` Phase 1 tables; `v2` adds `execution_plans`, `agents`, `tasks`, `task_transitions`, `evidence`, `claims`, `agent_messages`, `message_deliveries`, `review_verdicts`). Existing Phase 1 databases migrate cleanly without losing runs.
- **Immutable Claim History:** `claims` uses `PRIMARY KEY (claim_id, version)`; revisions append `version + 1` with `parent_claim_id`, never overwriting prior versions.
- **Cross-Run Isolation:** Database methods verify that agent, task, evidence, parent claim, dependency claim, and message references belong to the same `run_id`.
- **Known Verification Limitations (Phase 2):**
  - Claims remain `asserted` or `unresolved` in Phase 2 — never `supported` purely on model self-assertion.
  - `user_prompt` and `dependency_finding` evidence proposals check verbatim substring presence (`verified_in_input`), which confirms quotation existence only, not logical support.
  - `external_unverified` citations and `calculation` evidence remain `unverified` until Phase 3 adds deterministic verification tools, independent reviewers, and targeted revisions.

## 6. Planned Phase 3 Extensions

`NetworkRunner` exposes `ReviewHook` and the SQLite `review_verdicts` table as clean extension points for Phase 3's independent reviewer pool, deterministic calculation/fact verifiers, targeted claim-revision loop, and final Honesty Gate.
