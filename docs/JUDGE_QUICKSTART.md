# CrewForge — Judge Quickstart

CrewForge is a model-agnostic, self-assembling multi-agent network exposed via **Model Context Protocol (MCP)** (`solve`, `baseline`, `trace`) and observed through **CrewForge Mission Control** (`crewforge-dashboard`).

---

## 1. Prerequisites & Fresh Installation

- **Python:** `3.12` (`>=3.12,<3.13`)
- **Package manager:** `uv` (`https://docs.astral.sh/uv/`)

```bash
git clone https://github.com/Sushank37/CrewForge.git
cd CrewForge
uv sync --python 3.12
```

---

## 2. Local MCP vs. Remotely Hosted Assistants

- **Local MCP-compatible clients** (e.g., Claude Desktop, Cursor, Antigravity, local MCP Inspector) run on your machine and connect directly to `crewforge-mcp` over **stdio** or `http://127.0.0.1:8000/mcp` (**Streamable HTTP**).
- **Remotely hosted web assistants** (browser-only chat apps without a local MCP bridge) **cannot** reach `127.0.0.1` on your laptop directly. Do **not** expose an unauthenticated public tunnel to your local MCP server; use a local MCP client or the CLI (`crewforge solve` / `crewforge-dashboard`) instead.

---

## 3. Recommended Demonstration Configuration (Validated GPT-6 Luna Cloud Profile)

For live evaluation, use the validated **GPT-6 Luna** cloud reasoning profile. Keep `CREWFORGE_ALLOW_PAID_RUNS=false` permanently in `.env` and pass `CREWFORGE_ALLOW_PAID_RUNS=true` **only** in the temporary `crewforge-mcp` process environment.

> **Important:** Setting `CREWFORGE_CLOUD_REASONING_PROFILE=true` alone does **not** automatically apply every limit below; configure all variables explicitly.
> **Budget separation:** Judges should configure their own `OPENAI_API_KEY`, `CREWFORGE_MAX_COST_PER_RUN_USD`, and `CREWFORGE_TOTAL_BUDGET_CEILING_USD` (with `CREWFORGE_PROTECTED_RESERVE_USD=0.00` for an independent judge key). Do **not** override or draw down the team's existing `$4.00` protected reserve if running on the team's local `.crewforge/budget_authority.db`.

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export CREWFORGE_ALLOW_PAID_RUNS=true
export CREWFORGE_DB_PATH=".crewforge/demo_live.db"

# Model & reasoning profile
export CREWFORGE_PROVIDER=openai
export CREWFORGE_MODEL=gpt-6-luna
export CREWFORGE_OPENAI_REASONING_EFFORT=low
export CREWFORGE_CLOUD_REASONING_PROFILE=true
export CREWFORGE_SMALL_MODEL_PROFILE=false

# Token ceilings
export CREWFORGE_MAX_OUTPUT_TOKENS=8192
export CREWFORGE_PLANNER_MAX_OUTPUT_TOKENS=6144
export CREWFORGE_WORKER_MAX_OUTPUT_TOKENS=8192
export CREWFORGE_REVIEW_MAX_OUTPUT_TOKENS=6144
export CREWFORGE_REVISION_MAX_OUTPUT_TOKENS=8192
export CREWFORGE_SYNTHESIS_MAX_OUTPUT_TOKENS=6144

# Execution, review & safety bounds
export CREWFORGE_MAX_WORKERS=3
export CREWFORGE_MAX_TASKS=3
export CREWFORGE_MAX_CONCURRENCY=3
export CREWFORGE_MAX_COMMUNICATION_ROUNDS=1
export CREWFORGE_MAX_REVISION_ROUNDS=1
export CREWFORGE_MAX_MODEL_CALLS_PER_RUN=16
export CREWFORGE_MAX_RETRIES=0
export CREWFORGE_REQUEST_TIMEOUT_SECONDS=180
export CREWFORGE_RUN_DEADLINE_SECONDS=480
export CREWFORGE_ENABLE_HTTP_RETRIEVAL=false
export CREWFORGE_ENABLE_INDEPENDENT_REVIEW=true
export CREWFORGE_ENABLE_TARGETED_REPAIR=true
export CREWFORGE_ENABLE_PEER_MESSAGING=true

# Judge-configured independent budget limits
export CREWFORGE_MAX_COST_PER_RUN_USD=0.025
export CREWFORGE_TOTAL_BUDGET_CEILING_USD=1.00
export CREWFORGE_PROTECTED_RESERVE_USD=0.00
export CREWFORGE_PRICING_JSON='{"gpt-6-luna":{"input_per_1m":0.10,"output_per_1m":0.50}}'
```

### Experimental Local Ollama Profile (`qwen3:4b` — Not Recommended for Live Demo)
Local `ollama/qwen3:4b` (`CREWFORGE_OLLAMA_NUM_CTX=4096`, `CREWFORGE_OLLAMA_THINK=false`, `CREWFORGE_MAX_CONCURRENCY=1`) is **experimental and currently unreliable on 8 GB Apple Silicon M1 hardware**: in Phase 5G readiness testing (`canary_a`), `qwen3:4b` exceeded `max_tasks=3` on both planning and plan repair (`237.93 s` for 2 calls) and failed before specialist execution. Use the historical replay artifacts or the validated GPT-6 Luna cloud profile for judging.

---

## 4. MCP Client Configurations (Portable)

### Stdio MCP Client Configuration
```json
{
  "mcpServers": {
    "crewforge": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/CrewForge", "crewforge-mcp"],
      "env": {
        "CREWFORGE_DB_PATH": ".crewforge/demo_live.db"
      }
    }
  }
}
```

### Local Streamable HTTP MCP Server
```bash
uv run crewforge-mcp --transport streamable-http --host 127.0.0.1 --port 8000
```
Endpoint URL for local HTTP MCP clients: `http://127.0.0.1:8000/mcp`

---

## 5. Build & Start the React Interface + Mission Control

Build the React frontend and launch `crewforge-dashboard` (which requires the built React bundle and will not silently fall back to the legacy HTML view):

```bash
cd frontend && npm ci && npm run build && cd .. && uv run crewforge-dashboard --db-path .crewforge/judge_session.db --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765` in your browser. The server also exposes the shared MCP Streamable HTTP endpoint at `http://127.0.0.1:8765/mcp` using the same isolated judge session, credentials, and budget DB.

---

## 6. Invoke MCP Tools & Inspect in Mission Control

1. **`solve(prompt)`:** Runs the full CrewForge dynamic multi-agent pipeline (Planner → Specialists → Deterministic Arithmetic → Independent Reviewer → Claim Revision → Honesty Gate). Returns `run_id`, `status`, `answer`, and `verification`.
2. **`baseline(prompt)`:** Runs the single-agent baseline with equivalent deterministic tools for comparison.
3. **`trace(run_id)`:** Reads the persisted SQLite trace for `run_id` (**zero model inference**).
4. **Inspect in Mission Control:** Switch to **Live Monitor** (`Follow newest run` enabled) to watch live events, dynamic agent topology, claim versions, arithmetic grounding, and the Honesty Gate release decision.

---

## 7. Troubleshooting & Clean Shutdown

- **Paid run blocked:** Confirm `CREWFORGE_ALLOW_PAID_RUNS=true`, `OPENAI_API_KEY`, and `CREWFORGE_PRICING_JSON` are exported in the `crewforge-mcp` shell.
- **No runs in Live Monitor:** Ensure `crewforge-mcp` and `crewforge-dashboard --db-path` point to the exact same SQLite file (`.crewforge/demo_live.db`).
- **Clean shutdown:** Press `Ctrl+C` in both `crewforge-mcp` and `crewforge-dashboard` terminals. Verify `.env` retains `CREWFORGE_ALLOW_PAID_RUNS=false`.

---

## 8. Known Limitations

- **V2 Held-Out Benchmark:** In the 10-case same-model comparison (`gpt-6-luna`, ChatGPT-assisted blind semantic grading), single-agent baseline scored **96.50/100** vs CrewForge **55.50/100** (`1/10` network wins, `9` losses).
- **David vs Goliath:** Not executed — halted at the local `qwen3:4b` readiness canary.
- **Arithmetic Provenance Strictness:** Deterministic arithmetic requires verbatim digit/claim provenance; spelled-out quantities (e.g., `"five hours"`) or derived intermediate durations can trigger conservative Honesty Gate withholding even when calculations are numerically accurate.
- **Passing the Honesty Gate** verifies claim-level provenance and review, **not** factual correctness or complete requirement coverage.
