# CrewForge — Judge Quickstart

CrewForge is a model-agnostic, self-assembling multi-agent network exposed via **Model Context Protocol (MCP)** (`solve`, `baseline`, `trace`) and observed through **CrewForge Mission Control** (`crewforge-dashboard`).

---

## 1. Prerequisites & Fresh Installation

- **Python:** `3.12` (`>=3.12,<3.13`)
- **Package manager:** `uv` (`https://docs.astral.sh/uv/`)

```bash
git clone https://github.com/Sushank37/CrewForge.git
cd CrewForge
uv sync
```

---

## 2. Local MCP vs. Remotely Hosted Assistants

- **Local MCP-compatible clients** (e.g., Claude Desktop, Cursor, Antigravity, local MCP Inspector) run on your machine and can connect directly to `crewforge-mcp` over **stdio** or `http://127.0.0.1:8000/mcp` (**Streamable HTTP**).
- **Remotely hosted web assistants** (e.g., browser-only chat apps without a local MCP bridge) **cannot** reach `127.0.0.1` on your laptop directly. Do **not** expose an unauthenticated public tunnel to your local MCP server; use a local MCP client or the CLI (`crewforge solve` / `crewforge-dashboard`) instead.

---

## 3. Configure Provider, Model, API Key & Spending Limits

Copy `.env.example` (or create `.env`) in the repo root. Paid execution is **disabled by default** (`CREWFORGE_ALLOW_PAID_RUNS=false`).

### Option A: Local Zero-Cost Ollama (`qwen3:4b`)
```env
CREWFORGE_PROVIDER=ollama
CREWFORGE_MODEL=qwen3:4b
CREWFORGE_OLLAMA_BASE_URL=http://127.0.0.1:11434
CREWFORGE_OLLAMA_NUM_CTX=4096
CREWFORGE_OLLAMA_THINK=false
CREWFORGE_MAX_CONCURRENCY=1
CREWFORGE_MAX_RETRIES=0
CREWFORGE_ALLOW_PAID_RUNS=false
CREWFORGE_DB_PATH=.crewforge/demo_live.db
```

### Option B: Cloud Provider (e.g., OpenAI `gpt-6-luna`)
Provide your own key and **explicitly** authorize paid runs with strict caps:
```env
OPENAI_API_KEY=<YOUR_API_KEY>
CREWFORGE_PROVIDER=openai
CREWFORGE_MODEL=gpt-6-luna
CREWFORGE_OPENAI_REASONING_EFFORT=low
CREWFORGE_CLOUD_REASONING_PROFILE=true
CREWFORGE_ALLOW_PAID_RUNS=true
CREWFORGE_MAX_COST_PER_RUN_USD=0.025
CREWFORGE_TOTAL_BUDGET_CEILING_USD=1.00
CREWFORGE_PROTECTED_RESERVE_USD=0.00
CREWFORGE_DB_PATH=.crewforge/demo_live.db
```

> Use a dedicated demo trace database (`CREWFORGE_DB_PATH=.crewforge/demo_live.db`) so live demonstrations stay isolated from any other databases.

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
CREWFORGE_DB_PATH=.crewforge/demo_live.db uv run crewforge-mcp \
  --transport streamable-http --host 127.0.0.1 --port 8000
```
Endpoint URL for local HTTP MCP clients: `http://127.0.0.1:8000/mcp`

---

## 5. Start Mission Control Dashboard

In a second terminal, point Mission Control at the same demo database (read-only `mode=ro`):

```bash
uv run crewforge-dashboard --db-path .crewforge/demo_live.db --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765` in your browser.

---

## 6. Invoke MCP Tools & Inspect in Mission Control

1. **`solve(prompt)`:** Runs the full CrewForge dynamic multi-agent pipeline (Planner → Specialists → Deterministic Arithmetic → Independent Reviewer → Claim Revision → Honesty Gate). Returns `run_id`, `status`, `answer`, and `verification`.
2. **`baseline(prompt)`:** Runs the single-agent baseline with equivalent deterministic tools for comparison.
3. **`trace(run_id)`:** Reads the persisted SQLite trace for `run_id` (**zero model inference**).
4. **Inspect in Mission Control:** Switch to the **Live Monitor** tab (`Follow newest run` enabled) to watch events, dynamic agent topology, claim versions, arithmetic grounding, and the Honesty Gate release decision.

---

## 7. Troubleshooting & Clean Shutdown

- **Paid run blocked:** Verify `CREWFORGE_ALLOW_PAID_RUNS=true` and `CREWFORGE_MAX_COST_PER_RUN_USD` are set in the server process environment.
- **No runs in Live Monitor:** Ensure `crewforge-mcp` and `crewforge-dashboard --db-path` point to the exact same SQLite file (e.g., `.crewforge/demo_live.db`).
- **Clean shutdown:** Press `Ctrl+C` in the `crewforge-mcp` and `crewforge-dashboard` terminals. Leave `CREWFORGE_ALLOW_PAID_RUNS=false` in `.env` when done.

---

## 8. Known Limitations

- **V2 Held-Out Benchmark:** In the 10-case same-model comparison (`gpt-6-luna`, ChatGPT-assisted blind semantic grading), single-agent baseline scored **96.50/100** vs CrewForge **55.50/100** (`1/10` network wins, `9` losses), largely due to Honesty Gate withholding of solvable conclusions and document-grounding contradictions.
- **David vs Goliath (`qwen3:4b` vs `gpt-6.2-astra`):** Not executed — halted at the local readiness canary because `qwen3:4b` exceeded `max_tasks=3` on planning and plan repair.
- **Passing the Honesty Gate** verifies claim-level provenance and review, **not** factual correctness or complete requirement coverage.
