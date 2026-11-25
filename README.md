# CrewForge

CrewForge is a model-agnostic, self-assembling multi-agent intelligence network exposed through the Model Context Protocol (MCP), built by **Team Git Commit** for the **SEED Hackathon 2026 — Chain of Command** challenge.

## Implemented Functionality (Phase 1, Phase 2 & Phase 3)

- **Budget-Safe `ModelExecutor` & Shared MCP Execution:**
  - Single shared `asyncio.Semaphore` with deadline-bounded acquisition enforcing configured concurrency across baseline, planning, specialists, follow-ups, reviewers, targeted repairs, and synthesis (shared cleanly across concurrent MCP `solve` / `baseline` tool calls).
  - Every individual attempt acquires its own atomic pre-call SQLite `BEGIN IMMEDIATE` reservation and reconciles exactly once; hidden SDK/provider retries remain disabled when invoked through `ModelExecutor`.
  - Dispatched attempts count toward `max_model_calls_per_run` even when released as known-unbilled; unknown billing outcomes conservatively retain reserved cost.
  - Pre-call input token estimation incorporates system prompts, task context, and serialized structured-output JSON schemas; credentials and raw provider errors are sanitized before persistence.
- **Dynamic Lead Planner (`LeadPlanner`) & DAG Scheduler (`DAGScheduler`):** Decomposes arbitrary prompts into a validated `ExecutionPlan` with prompt-specific specialist roles and acyclic dependencies (`effective_max_workers=3`, `effective_max_tasks=3` for local Ollama), executes independent branches concurrently, enforces deadline/call-budget reservations (`optional_stage_skipped`), and transitions failed dependents cleanly to `blocked`.
- **Claim Contracts, Provenance & Safe Arithmetic Tool (`DeterministicArithmeticTool`):**
  - Explicit `ClaimCategory` (`given_fact`, `external_factual_assertion`, `computation`, `logical_inference`, `assumption`, `recommendation`, `uncertainty`) and separate `VerificationStatus` (`provenance_verified`, `computationally_verified`, `independently_reviewed_inference`, `unsupported`, `contradicted`, `stale_pending_review`).
  - Safe `Decimal` + allowlisted AST calculator (`src/crewforge/tools/arithmetic.py`, no `eval`/`exec`) persists immutable `tool_executions` records, checks numeric/provenance input grounding, and supports fair baseline tool parity (`BaselineRunner`).
  - Deterministic provenance verification (`evidence_verifier.py`) checks verbatim quotation, numeric consistency, and surrounding negation polarity for `user_prompt` / `dependency_finding` excerpts while keeping unretrieved `external_unverified` citations as `unverified` (`UnavailableExternalRetriever`).
- **Independent Adversarial Review (`ReviewerPool`), `ClaimDependencyGraph` & `TargetedRepairEngine`:**
  - Enforces strict author/reviewer separation (`reviewer_id != author_agent_id`) in code and SQLite, running deterministic prechecks plus compact/separate model review; omitted verdicts deterministically mark claims `unresolved`.
  - Blocking objections are routed back via `MessageBus` to the originating specialist for explicit `(claim_id, expected_version)` targeted revision (never positional replacement), preserving rejected prior versions, transitively marking downstream dependents `stale`, and re-reviewing revised claims.
- **Evidence-Constrained `LeadSynthesizer` & Deterministic `HonestyGate`:** Consumes Lead inbox messages (`consumed` only after model invocation), requires versioned claim citations on released assertions, blocks rejected/stale/unreviewed claims and unverified new factual text on both full and partial release paths, and persists `honesty_gate_decisions` + `VerificationSummary`.
- **MCP Server (`FastMCP`) & CLI:** Exposes `solve(prompt)`, `baseline(prompt)`, and `trace(run_id)` over `stdio` (`stderr`-only logging) and localhost `streamable-http`.

### Planned / Remaining Before Final Submission

- Optional external `EvidenceRetriever` backend if an authorized retrieval source is configured (internet fetching remains disabled by default via `UnavailableExternalRetriever`).
- Final pre-submission David vs Goliath evaluation comparing the local `qwen3:4b` network against a stronger cloud baseline within the untouched \$6.10 OpenAI budget cap.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/ROADMAP.md](docs/ROADMAP.md) for full details.

## Installation

Requires **Python 3.12** and **uv**.

```bash
uv sync --python 3.12
```

## Local Ollama Setup (Apple Silicon 8 GB Profile)

1. Pull the local candidate model (already installed if `ollama list` shows `qwen3:4b`):

```bash
ollama pull qwen3:4b
```

2. Copy `.env.example` to `.env` for local defaults (`qwen3:4b`, `CREWFORGE_MAX_CONCURRENCY=1`, `CREWFORGE_MAX_RETRIES=0`, `CREWFORGE_OLLAMA_NUM_CTX=4096`, `CREWFORGE_OLLAMA_THINK=false`):

```bash
cp .env.example .env
```

> **Note:** On `qwen3` models in Ollama, `CREWFORGE_OLLAMA_THINK=false` disables the separate thinking trace via Ollama's top-level `think` parameter so structured JSON outputs complete faster within tight token/memory budgets. Leave `CREWFORGE_OLLAMA_THINK` unset for models that do not support `think`.

## Configuration Reference

| Variable | Default | Description |
| --- | --- | --- |
| `CREWFORGE_PROVIDER` | `ollama` | Active provider (`ollama` or `openai`) |
| `CREWFORGE_MODEL` | `qwen3:4b` | Model identifier |
| `CREWFORGE_OPENAI_API_KEY` | unset | OpenAI API key (also falls back to `OPENAI_API_KEY`) |
| `CREWFORGE_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Local Ollama endpoint |
| `CREWFORGE_OLLAMA_NUM_CTX` | `4096` | Ollama context window (`options.num_ctx`) |
| `CREWFORGE_OLLAMA_THINK` | unset (`false` in `.env.example`) | Optional top-level Ollama `think` parameter |
| `CREWFORGE_REQUEST_TIMEOUT_SECONDS` | `60.0` (`180.0` in local `.env`) | Per-attempt request timeout |
| `CREWFORGE_RUN_DEADLINE_SECONDS` | `300.0` (`480.0` in local `.env`) | Total wall-clock deadline per run |
| `CREWFORGE_MAX_RETRIES` | `0` | Bounded retry count owned by `ModelExecutor` |
| `CREWFORGE_MAX_CONCURRENCY` | `3` (`1` auto-default for Ollama) | Shared concurrency semaphore limit |
| `CREWFORGE_MAX_OUTPUT_TOKENS` | `1024` | Output token cap per call (operation caps configurable for planner/worker/review/revision/synthesis) |
| `CREWFORGE_MAX_WORKERS` / `CREWFORGE_MAX_TASKS` | `4` / `6` (`3` / `3` auto-cap for Ollama) | Planner team and DAG size bounds |
| `CREWFORGE_MAX_COMMUNICATION_ROUNDS` | `1` | Max peer follow-up rounds after DAG tasks |
| `CREWFORGE_MAX_REVISION_ROUNDS` | `1` | Max targeted claim-repair rounds after review |
| `CREWFORGE_SEPARATE_REVIEW_DIMENSIONS` | `false` | `false` runs compact combined reviewer pass; `true` splits evidence/consistency passes |
| `CREWFORGE_MAX_MODEL_CALLS_PER_RUN` | `16` | Maximum dispatched model attempts per run |
| `CREWFORGE_MAX_COST_PER_RUN_USD` | `0.05` | Maximum estimated cloud cost per run (USD) |
| `CREWFORGE_MAX_CUMULATIVE_COST_USD` | `1.00` | Maximum cumulative expenditure across runs (USD) |
| `CREWFORGE_PRICING_JSON` | `{}` | JSON map of model -> `{"input_per_1m": float, "output_per_1m": float}` |
| `CREWFORGE_DB_PATH` | `.crewforge/crewforge.db` | SQLite database file path |

> **Budget Disclaimer:** Application-level cost estimation and pre-call reservations are software safeguards based on configured pricing and token bounds. They are **not** a guaranteed provider billing cap; always configure hard spending limits directly in your cloud provider dashboard.

## Testing & Linting

All ordinary tests run completely offline without paid API calls:

```bash
uv run pytest
uv run ruff check .
```

### Real-Provider Smoke Test (Local Ollama)

```bash
uv run crewforge smoke-test --provider ollama --model qwen3:4b --mode baseline
uv run crewforge smoke-test --provider ollama --model qwen3:4b --mode network
```

## Exact CLI Commands

```bash
uv run crewforge solve "Your prompt here"
uv run crewforge baseline "Your prompt here"
uv run crewforge trace <run_id>
uv run crewforge budget
```

## Exact MCP Commands & Client Configuration

Launch over `stdio` (`stdout` strictly reserved for MCP JSON-RPC; logs go to `stderr`):

```bash
uv run crewforge-mcp
```

Launch over localhost Streamable HTTP:

```bash
uv run crewforge-mcp --transport streamable-http --host 127.0.0.1 --port 8000
```

**Stdio MCP Client Configuration:**

```json
{
  "mcpServers": {
    "crewforge": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/Users/sushanksingh/Desktop/CrewForge",
        "crewforge-mcp"
      ],
      "env": {
        "CREWFORGE_PROVIDER": "ollama",
        "CREWFORGE_MODEL": "qwen3:4b",
        "CREWFORGE_OLLAMA_BASE_URL": "http://127.0.0.1:11434",
        "CREWFORGE_OLLAMA_NUM_CTX": "4096",
        "CREWFORGE_OLLAMA_THINK": "false",
        "CREWFORGE_MAX_CONCURRENCY": "1",
        "CREWFORGE_MAX_RETRIES": "0"
      }
    }
  }
}
```
