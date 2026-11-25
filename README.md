# CrewForge

CrewForge is a model-agnostic, self-assembling multi-agent intelligence network exposed through the Model Context Protocol (MCP), built by **Team Git Commit** for the **SEED Hackathon 2026 — Chain of Command** challenge.

## Implemented Functionality (Phases 1–4)

- **Budget-Safe `ModelExecutor`, Cloud Guard & Shared MCP Execution:**
  - Single shared `asyncio.Semaphore` with deadline-bounded acquisition enforces configured concurrency (`1` local Ollama default, `3` cloud OpenAI) across concurrent MCP `solve` / `baseline` tool runs.
  - Explicit `CREWFORGE_ALLOW_PAID_RUNS=false` fail-closed authorization guard blocks paid API execution unless authorized; paid cloud runs automatically persist to `CREWFORGE_CLOUD_DB_PATH` (`.crewforge/cloud_dev.db`), require non-empty pricing in `CREWFORGE_PRICING_JSON`, reserve every attempt atomically (`BEGIN IMMEDIATE`), and record actual `returned_model` identities.
- **Claim-Locked Honesty Gate & Anti-Laundering (`core/honesty_gate.py`):**
  - Default final answers render verified factual text directly from exact approved ledger `Claim` versions — forbidding Lead paraphrases, negation reversals, unreferenced factual prose, or laundering through `conditional_conclusions`, `recommendations`, `assumptions`, `missing_information`, `unresolved_issues`, `confidence_notes`, or raw task error messages on both full and partial paths.
  - Rejected claim statements stay preserved in `trace.claims` and are never echoed unnecessarily in rendered answers.
- **Exact Arithmetic Provenance Lineage (`tools/arithmetic.py`):**
  - `DeterministicArithmeticTool` resolves every variable to an exact target (`up_1..up_N` prompt spans, `doc_1..doc_N` verified evidence excerpts, current `SUPPORTED` claim versions, prior grounded `ToolExecutionRecord`s, or explicit `const:` / `assumption:` references) — rejecting swapped variables, fabricated targets, unrelated sentence numbers, ungrounded inline AST constants, unit mismatches, and stale claim versions.
- **Strict Reviewer Verdict Validation (`core/reviewer.py`):**
  - Enforces at most one verdict per `(claim_id, claim_version, reviewer_id, review_dimension)` in a review round, strips cross-claim evidence refs, keeps skipped/missing dimensions explicitly incomplete (`UNRESOLVED`), triggers disputes on conflicts, and rechecks fresh DB premise currency before approval.
- **Honest Requirement Coverage (`VerificationSummary`):**
  - Tracks `req_1..req_K` linkage separately across `answered`, `partially_answered`, `unresolved`, and `missing`, and exposes distinct `execution_completed`, `release_checks_executed`, `content_released_safely`, `coverage_methodology`, and `independent_grade` (`None` internally).
- **External Retrieval Foundation (`core/retrieval.py`):**
  - `LocalDocumentRetriever` ingests operator-supplied UTF-8 documents (`--doc` / `CREWFORGE_REFERENCE_DOCUMENTS`) into immutable verified `Evidence` with SHA-256 `content_hash`, UTC `retrieved_at`, and character-offset `source_location` for both Baseline and Network.
  - `AllowlistedHttpRetriever` (disabled by default via `CREWFORGE_ENABLE_HTTP_RETRIEVAL=false`) enforces HTTPS-only explicit host allowlists, public-IP SSRF/DNS-rebinding checks, `follow_redirects=False`, response size/type limits, and credential isolation. No open-web search backend exists in Phase 4.
- **Provider Portability (`openai`, `openai_compatible`, `ollama`):**
  - Explicit `CREWFORGE_OPENAI_REASONING_EFFORT` support (`none`, `low`, `medium`, `high`, `xhigh`, `max`) omits incompatible `temperature` on reasoning models (`gpt-6-luna`); `openai_compatible` uses operator-supplied `CREWFORGE_OPENAI_BASE_URL` via Chat Completions with fail-closed pricing. (Anthropic/Gemini protocols are not implemented.)
- **Adaptive Execution & Strong Fair Baselines:**
  - Concise claim/requirement aliases (`c1`, `up_1`, `req_1`), bounded claims/worker, selective follow-ups, and a Gate-protected deterministic synthesis fast path (`CREWFORGE_ENABLE_DETERMINISTIC_SYNTHESIS_FAST_PATH=true`) skip redundant synthesis calls when all requirements are covered cleanly.
  - Upgraded `BaselineRunner` supports deterministic arithmetic parity, equal document ingestion (`--doc`), and optional `--compute-matched` single-agent self-check refinement without spawning specialists or reviewers.
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
