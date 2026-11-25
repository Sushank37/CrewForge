# CrewForge

CrewForge is a model-agnostic, self-assembling multi-agent intelligence network exposed through the Model Context Protocol (MCP), built by **Team Git Commit** for the **SEED Hackathon 2026 — Chain of Command** challenge.

## Phase 1 Status

Phase 1 delivers the verified engineering foundation:

- **Typed Data Contracts (Pydantic v2):** `AgentSpec`, `TaskSpec`, `ExecutionPlan`, `Claim`, `Evidence`, `AgentMessage`, `ReviewVerdict`, `RunConfig`, `RunResult`, `UsageMetrics`, `ExecutionEvent`, and `RunTrace`.
- **Model Provider Abstraction:** Asynchronous `ModelProvider` interface with text generation, Pydantic-validated structured JSON generation, token accounting, timeouts, bounded retries, cancellation handling, and structured errors.
  - `OpenAIProvider` backed by the official `AsyncOpenAI` Responses API (`responses.create` / `responses.parse`).
  - `OllamaProvider` backed by Ollama's local `/api/chat` HTTP API (initial local candidate: `qwen3:4b`).
- **Concurrency-Safe Persistent Budget Manager:** Atomic SQLite (`BEGIN IMMEDIATE`) pre-call cost reservations, per-run and cumulative USD caps, model-call caps, fail-closed enforcement when paid-model pricing is unknown, retry accounting, and conservative handling of unknown billing outcomes.
- **SQLite Persistence:** Run-isolated storage for run metadata, sequential execution events, token usage records, and budget reservations.
- **Single-Agent Baseline:** Asynchronous `BaselineRunner` producing structured answers that explicitly identify missing information, assumptions, token usage, and latency.
- **MCP Server:** Official MCP Python SDK (`FastMCP`) exposing `baseline(prompt)` and `trace(run_id)` over `stdio` (default) and localhost `streamable-http`, with operational logs directed strictly to `stderr`.

### Planned Features (Phase 2+)

- Lead agent dynamic task decomposition and specialist creation (`solve(prompt)`)
- Dependency-aware concurrent task scheduler and shared `AgentMessage` bus
- Versioned claim ledger with evidence provenance (`asserted`, `supported`, `rejected`, `unresolved`)
- Independent reviewer pool and targeted revision loop
- Final Honesty Gate and David vs Goliath evaluation harness

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/ROADMAP.md](docs/ROADMAP.md) for details.

## Installation

Requires **Python 3.12** and **uv**.

```bash
uv sync --python 3.12
```

## Environment Setup & Configuration

Copy `.env.example` to `.env` and adjust values as needed (never commit `.env` or real API keys):

```bash
cp .env.example .env
```

Key environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `CREWFORGE_PROVIDER` | `ollama` | Active provider (`ollama` or `openai`) |
| `CREWFORGE_MODEL` | `qwen3:4b` | Model identifier |
| `CREWFORGE_OPENAI_API_KEY` | unset | OpenAI API key (also falls back to `OPENAI_API_KEY`) |
| `CREWFORGE_OPENAI_BASE_URL` | unset | Optional OpenAI-compatible base URL |
| `CREWFORGE_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Local Ollama endpoint |
| `CREWFORGE_REQUEST_TIMEOUT_SECONDS` | `60.0` | Per-attempt request timeout |
| `CREWFORGE_MAX_RETRIES` | `2` | Bounded retry count on retryable errors |
| `CREWFORGE_MAX_CONCURRENCY` | `3` | Maximum concurrent model calls |
| `CREWFORGE_MAX_OUTPUT_TOKENS` | `1024` | Output token cap per call |
| `CREWFORGE_MAX_MODEL_CALLS_PER_RUN` | `16` | Maximum model calls per run |
| `CREWFORGE_MAX_COST_PER_RUN_USD` | `0.05` | Maximum estimated cloud cost per run (USD) |
| `CREWFORGE_MAX_CUMULATIVE_COST_USD` | `1.00` | Maximum cumulative expenditure across runs (USD) |
| `CREWFORGE_PRICING_JSON` | `{}` | JSON map of model name to `{"input_per_1m": float, "output_per_1m": float}` |
| `CREWFORGE_DB_PATH` | `.crewforge/crewforge.db` | SQLite database file path |
| `CREWFORGE_LOG_LEVEL` | `INFO` | Logging verbosity (written to `stderr` only) |

> **Budget Disclaimer:** Application-level cost estimation and pre-call reservations are software safeguards based on configured pricing and token bounds. They are **not** a guaranteed provider billing cap; always configure hard spending limits directly in your cloud provider dashboard. Paid models without an entry in `CREWFORGE_PRICING_JSON` fail closed before any API request is made.

## Testing & Linting

All ordinary tests run completely offline without paid API calls:

```bash
uv run pytest
uv run ruff check .
```

### Real-Provider Smoke Test (Manual Only)

A single bounded real-provider smoke test command is available when you are ready to verify live connectivity (do not run unattended against paid accounts without setting `CREWFORGE_PRICING_JSON`):

```bash
# Local Ollama candidate
uv run crewforge smoke-test --provider ollama --model qwen3:4b

# OpenAI (requires CREWFORGE_OPENAI_API_KEY and CREWFORGE_PRICING_JSON)
uv run crewforge smoke-test --provider openai --model <configured-model>
```

## CLI Usage

```bash
uv run crewforge baseline "Explain the trade-offs between WAL and rollback journals in SQLite."
uv run crewforge trace <run_id>
uv run crewforge budget
```

## MCP Server

Launch with `stdio` transport (recommended for local MCP clients; `stdout` is reserved strictly for JSON-RPC frames):

```bash
uv run crewforge-mcp
```

Launch with localhost Streamable HTTP transport:

```bash
uv run crewforge-mcp --transport streamable-http --host 127.0.0.1 --port 8000
```

### Reproducible MCP Client Configuration

**Stdio client config (`mcpServers`):**

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
        "CREWFORGE_OLLAMA_BASE_URL": "http://127.0.0.1:11434"
      }
    }
  }
}
```

**Streamable HTTP client endpoint:**

`http://127.0.0.1:8000/mcp` (after starting `uv run crewforge-mcp --transport streamable-http --host 127.0.0.1 --port 8000`).
