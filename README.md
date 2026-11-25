# CrewForge

CrewForge is a model-agnostic, self-assembling multi-agent intelligence network exposed through the Model Context Protocol (MCP), built by **Team Git Commit** for the **SEED Hackathon 2026 — Chain of Command** challenge.

## Implemented Functionality (Phases 1–5 Engine Freeze)

- **Unified Persistent Budget Authority & Reserve Protection (`core/budget.py`):**
  - Standalone canonical ledger (`.crewforge/budget_authority.db`) reconciles historical spending from `.crewforge/crewforge.db` and `.crewforge/cloud_dev.db` without double-counting ($0.025516 committed across pilot development).
  - Enforces the hackathon-mandated $4.00 emergency protected reserve (from $6.10 total OpenAI balance, leaving $2.10 total spendable ceiling) and category caps (`phase5_same_model` max $0.75, `david_vs_goliath` max $1.10).
  - Conservative unknown-outcome billing and immediate atomic reservations (`BEGIN IMMEDIATE`) ensure strict compliance.
- **Reproducible Blind Benchmark & Scoring Harness (`evaluation/`, `benchmarks/`):**
  - Cryptographically random per-experiment blinding seed (`secrets.token_hex(32)`) persisted separately in `blind_key.json`; strips multi-claim tags, run IDs, model names, and CrewForge verification metadata from `confidence_notes` while preserving assumptions, missing info, and full reference documents.
  - Separates deterministic diagnostics from independent semantic grading using the standardized 100-point rubric (`30/25/20/15/10` with `-5` per unsupported material assertion, max `-20`, floor `0`). Ungraded runs report `pending independent grading`.
  - Immutable experiment fingerprints isolate cached runs by manifest, prompt, document hashes, git commit, provider, model, baseline mode, ablation variant, and runtime limits.
  - Real evaluation-only ablation switches (`full`, `no_reviewer`, `no_repair`, `no_peer_messages`) and asymmetric David (`ollama/qwen3:4b`) vs. Goliath (`openai/gpt-6.2-astra`) configuration.
- **Budget-Matched Iterative Baseline (`core/baseline.py`):**
  - Configurable call and output-token allowances (`baseline_max_calls`, `baseline_max_output_tokens`) with up to 3 self-critique rounds, full reference-document access, verified tool incorporation, and honest reporting of unused compute.
- **Safe Portable Demo Trace Export (`core/demo_trace.py`):**
  - Defaults `include_prompt=False`, recursively sanitizes credentials/headers/nested prompt leaks, includes evidence provenance, and preserves actual historical timestamps and outcomes.
- **Small-Model Profile & Deterministic Arithmetic Parity:**
  - Compact profile for Apple Silicon M1 (concurrency 1, max 2 workers, max 3 tasks, ~4096 context).
  - Parity arithmetic tool resolving variables to exact prompt/evidence spans with AST literal verification.
- **Reasoning-Aware Cloud Profile & Failed-Call Reconciliation (`providers/openai_provider.py`, `core/budget.py`):**
  - Explicit `CrewForgeSettings.for_cloud_reasoning_profile()` (`8192` global/worker/revision, `6144` planner/review/synthesis) prevents reasoning-token starvation without clamping operation-specific limits or altering Ollama's local profile.
  - `OpenAIProvider.generate_structured` uses `responses.create` with strict JSON Schema (`to_strict_json_schema`) to capture `provider_response_id`, `response_status`, `incomplete_reason`, `reasoning_tokens`, and `cached_input_tokens` before validation.
  - Failed structured calls with known usage (e.g., `incomplete` at `max_output_tokens`, refusal, or `ValidationError`) reconcile exact cost (`status="reconciled"`, `outcome="failed"`) without incrementing `successful_operations`.
  - Accurate `stages_executed` and `stage_telemetry` reflect persisted work rather than feature flags.
- **MCP Server (`FastMCP`) & CLI:** Exposes `solve(prompt)`, `baseline(prompt)`, and `trace(run_id)` over `stdio` (`stderr`-only logging) and localhost `streamable-http`.

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
| `CREWFORGE_PROVIDER` | `ollama` | Active provider (`ollama`, `openai`, or `openai_compatible`) |
| `CREWFORGE_MODEL` | `qwen3:4b` | Model identifier (cloud pilot candidate: `gpt-6-luna`) |
| `CREWFORGE_OPENAI_API_KEY` | unset | OpenAI API key (also falls back to `OPENAI_API_KEY`; never exposed in prompts/traces) |
| `CREWFORGE_OPENAI_BASE_URL` | unset | Operator-controlled endpoint override / `openai_compatible` server URL |
| `CREWFORGE_OPENAI_REASONING_EFFORT` | unset (`low` for pilots) | OpenAI reasoning effort (`none`, `low`, `medium`, `high`, `xhigh`, `max`) |
| `CREWFORGE_ALLOW_PAID_RUNS` | `false` | Explicit fail-closed guard required to permit paid provider calls |
| `CREWFORGE_ENABLE_HTTP_RETRIEVAL` | `false` | Optional allowlisted HTTPS retriever toggle (disabled by default) |
| `CREWFORGE_ALLOWED_RETRIEVAL_HOSTS` | `[]` | Exact HTTPS host allowlist when HTTP retrieval is enabled |
| `CREWFORGE_REFERENCE_DOCUMENTS` | `[]` | Optional list/paths of operator-supplied reference documents |
| `CREWFORGE_BASELINE_COMPUTE_MATCHED` | `false` | Enables iterative single-agent self-check pass in `BaselineRunner` |
| `CREWFORGE_ENABLE_DETERMINISTIC_SYNTHESIS_FAST_PATH` | `true` | Skips synthesis LLM call cleanly when Gate-approved claims cover requirements |
| `CREWFORGE_CLOUD_DB_PATH` | `.crewforge/cloud_dev.db` | Persistent SQLite budget & trace DB used automatically for cloud runs |
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
uv run crewforge solve "Your prompt here" [--doc path/to/doc.md]
uv run crewforge baseline "Your prompt here" [--doc path/to/doc.md] [--compute-matched]
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
