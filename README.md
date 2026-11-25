# CrewForge

CrewForge is a model-agnostic, self-assembling multi-agent intelligence network exposed through the Model Context Protocol (MCP), built by **Team Git Commit** for the **SEED Hackathon 2026 — Chain of Command** challenge.

## Implemented Functionality (Phase 1 & Phase 2)

- **Budget-Safe `ModelExecutor` & Corrected Accounting:**
  - Single shared `asyncio.Semaphore` enforcing configured concurrency across baseline, planning, specialists, follow-ups, and synthesis.
  - Every individual attempt acquires its own atomic pre-call SQLite `BEGIN IMMEDIATE` reservation and reconciles exactly once.
  - Hidden SDK/provider retries are disabled when invoked through `ModelExecutor`; retry orchestration is owned by `ModelExecutor`.
  - Dispatched attempts count toward `max_model_calls_per_run` even when released as known-unbilled; unknown billing outcomes (e.g., timeouts or missing paid usage) conservatively retain their reserved cost.
  - Pre-call input token estimation incorporates system prompts, task context, and serialized structured-output JSON schemas.
  - Raw provider errors and credentials are sanitized before persistence.
- **Dynamic Lead Planner (`LeadPlanner`):** Decomposes arbitrary prompts into a validated `ExecutionPlan` with prompt-specific specialist roles, DAG dependencies, acceptance criteria, and deliverables (no hard-coded domain routing or role lists), allowing at most one budgeted plan-repair attempt.
- **Dependency-Aware Async Scheduler (`DAGScheduler`):** Executes independent tasks concurrently up to configured concurrency, transitions tasks through `pending -> running -> completed | failed | blocked`, prevents deadlocks on dependency failure, and enforces a total `run_deadline_seconds` alongside per-request timeouts.
- **Specialist Execution & Provenance Ledger (`SpecialistWorker`, `ClaimLedger`):** Produces concise findings, immutable versioned `Claim`s (`asserted` / `unresolved` — never `supported` on self-assertion alone), and provenance-checked `Evidence` records (`user_prompt` and `dependency_finding` substring verification; `external_unverified` and `calculation` kept `unverified`).
- **Real Peer-to-Peer Messaging (`MessageBus`):** Run-scoped direct and broadcast `AgentMessage` persistence with per-recipient delivery/consumption tracking (`pending -> delivered -> consumed`), feeding messages into dependent tasks and one bounded communication follow-up round.
- **Lead Synthesis (`LeadSynthesizer`) & `NetworkRunner`:** Synthesizes specialist findings within reserved synthesis call headroom, validates all referenced `claim_id`s belong to the current run, returns explicit `partial` results on budget/deadline exhaustion, and surfaces Phase 2 verification limits honestly.
- **MCP Server (`FastMCP`) & CLI:** Exposes `solve(prompt)`, `baseline(prompt)`, and `trace(run_id)` over `stdio` (with `stderr`-only operational logging) and localhost `streamable-http`.

### Planned Functionality (Phase 3)

- Independent reviewer pool (`ReviewVerdict` generation) checking unsupported claims, calculation accuracy, and cross-claim contradictions
- Targeted revision loop returning rejected claims to originating specialists
- Deterministic verification tools & full Honesty Gate enforcement
- David vs Goliath benchmark evaluation (`qwen3:4b` network vs single-agent baseline)

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
| `CREWFORGE_REQUEST_TIMEOUT_SECONDS` | `60.0` | Per-attempt request timeout |
| `CREWFORGE_RUN_DEADLINE_SECONDS` | `300.0` | Total wall-clock deadline per run |
| `CREWFORGE_MAX_RETRIES` | `0` | Bounded retry count owned by `ModelExecutor` |
| `CREWFORGE_MAX_CONCURRENCY` | `3` (`1` auto-default for Ollama) | Shared concurrency semaphore limit |
| `CREWFORGE_MAX_OUTPUT_TOKENS` | `1024` | Output token cap per call |
| `CREWFORGE_MAX_WORKERS` / `CREWFORGE_MAX_TASKS` | `4` / `6` | Planner team and DAG size bounds |
| `CREWFORGE_MAX_COMMUNICATION_ROUNDS` | `1` | Max peer follow-up rounds after DAG tasks |
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
