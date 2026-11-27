# CrewForge Public Cloud MCP Deployment Guide (SEED Hackathon 2026)

This guide covers deploying the dedicated **unauthenticated public evaluation MCP service** (`crewforge-public-mcp`) to a public cloud host with HTTPS at `/mcp` — with **no login, password, API key, token, VPN, tunnel, or IP/country restrictions** for judges.

> **Important:** The public evaluation service (`crewforge-public-mcp`) is completely separate from the authenticated browser application (`crewforge-dashboard`). Never deploy the private browser workspace or local `.crewforge/` databases to the public evaluation container.

---

## 1. Architecture & Isolation

- **Endpoint:** `https://<your-cloud-host>/mcp` (Streamable HTTP) and `https://<your-cloud-host>/health`
- **Single Replica + Persistent Storage:** Deploy as **1 replica (`replicas = 1`)** with a persistent volume mounted at `/data` so SQLite WAL job/budget state and in-memory background tasks stay consistent without cross-replica split-brain.
- **No Judge Auth Required:** Judges connect using only the public HTTPS URL.
- **Anonymous Public Evaluation Notice:** Evaluation tasks submitted to `/mcp` are not private personal workspaces. Do not submit confidential personal data.
- **Resource & Queue Bounds:** Requests are capped at `512 KB`, prompts at `32,000` chars, reference documents at `20` docs / `200,000` chars, concurrent jobs via `CREWFORGE_PUBLIC_MAX_CONCURRENT_JOBS` (default `4`), and pending queue size via `CREWFORGE_PUBLIC_MAX_PENDING_JOBS` (default `12`, returning retryable `capacity_exhausted` errors when full). Both `run_task` and `run_single_agent` support optional `idempotency_key`.
- **Server-Side Inference & Budget Protection:**
  - Model provider, model name, API key, and spending caps are configured **only** via server-side environment secrets.
  - Callers cannot supply arbitrary `openai_base_url` / `ollama_base_url` endpoints or access private networks.
  - Task IDs use 128-bit random tokens (`tsk_pub_<32-hex>`), and no task-listing endpoint is exposed.
  - Interrupted in-flight jobs are marked `failed` on server restart so duplicate paid inference is never triggered automatically.

---

## 2. Required Server-Side Environment Variables

Set these in your cloud platform's Secrets / Environment UI (never commit secrets to git):

```bash
# Storage (mount a persistent volume at /data)
CREWFORGE_PUBLIC_MCP_DB_PATH=/data/public_mcp.db
CREWFORGE_PUBLIC_MCP_BUDGET_DB_PATH=/data/public_mcp_budget.db
CREWFORGE_PUBLIC_MAX_CONCURRENT_JOBS=4

# Server-side provider configuration (example: Groq free tier or OpenAI)
CREWFORGE_PROVIDER=openai_compatible
CREWFORGE_OPENAI_BASE_URL=https://api.groq.com/openai/v1
CREWFORGE_MODEL=openai/gpt-oss-120b
CREWFORGE_OPENAI_API_KEY=<YOUR_SERVER_SIDE_API_KEY>

# Spending safeguards (must be explicitly true in cloud deployment; keep false in local dev)
CREWFORGE_ALLOW_PAID_RUNS=true
CREWFORGE_MAX_COST_PER_RUN_USD=0.05
CREWFORGE_MAX_CUMULATIVE_COST_USD=2.00
CREWFORGE_PROTECTED_RESERVE_USD=0.00
CREWFORGE_MAX_MODEL_CALLS_PER_RUN=16
```

---

## 3. Verified Free / Student-Credit Hosting Options

1. **Railway (Hobby / Free Trial or GitHub Student Developer Pack credits)**
   - **Pros:** Native `Dockerfile` deployment, automatic public HTTPS (`*.up.railway.app`), persistent **Railway Volumes** (mount a volume at `/data` in the service settings — do **not** declare `VOLUME` in `Dockerfile`), long-running background asyncio tasks survive client disconnects, clean `PORT` injection.
   - **Storage & Permissions:** Attach a Railway Volume at `/data` and keep **1 replica**. On container start, the entrypoint ensures `/data` is writable by the non-root `crewforge` user (UID `10001`) before dropping privileges and launching `crewforge-public-mcp`.
   - **Limitations:** Free trial has a monthly credit cap; persistent volumes require a non-ephemeral plan/credit balance.

2. **Fly.io (Pay-as-you-go free allowance / Student credits)**
   - **Pros:** Global anycast HTTPS (`*.fly.dev`), persistent Fly Volumes (`fly volumes create data --size 1` mounted at `/data`), no idle sleep when `auto_stop_machines = false`, full `Dockerfile` support.
   - **Limitations:** Requires a payment method on file even within free resource allowances.

3. **Render (Free Web Service or Student credits)**
   - **Pros:** Zero-config `Dockerfile` deploy, automatic HTTPS (`*.onrender.com`), `/health` check support.
   - **Limitations:** Free tier spins down after 15 minutes of inactivity (causing cold starts and wiping ephemeral disk unless a paid Persistent Disk is attached). Background tasks are interrupted on spin-down.

4. **Google Cloud Run (Free tier / Student Cloud credits)**
   - **Pros:** Automatic HTTPS, generous monthly free tier, clean container deployment.
   - **Limitations:** Cloud Run is request-scoped by default (CPU may be throttled after HTTP response returns unless `--no-cpu-throttling` / always-allocated CPU is enabled), and local container filesystem is ephemeral (requires Cloud Storage / external DB for persistence across revisions).

---

## 4. Final External MCP Verification Procedure (Public URL Only)

Once deployed, test from an **external MCP-compatible AI assistant** (e.g., Claude Desktop, Cursor, ChatGPT/MCP connector, or MCP Inspector) configured with **only** the public URL:

```json
{
  "mcpServers": {
    "crewforge-public": {
      "url": "https://<your-cloud-host>/mcp"
    }
  }
}
```

Verify in order:
1. **Health check:** `GET https://<your-cloud-host>/health` returns `200 OK` with `"status": "ok"`.
2. **Discovery:** Client connects with zero auth headers and discovers all 6 tools: `run_task`, `get_task_status`, `get_task_result`, `get_task_trace`, `run_single_agent`, `cancel_task`.
3. **Async multi-agent submission:** Call `run_task(prompt=..., reference_documents=[...])` -> returns `task_id` immediately (`status: "queued"`).
4. **Polling & completion:** Poll `get_task_status(task_id)` until `status` is `"completed"` or `"partial"`.
5. **Result & sources:** Call `get_task_result(task_id)` -> verify `answer`, `sources`, `evidence`, `verification` (`honesty_gate_passed`, `release_mode`), and `unresolved_requirements`.
6. **Trace inspection:** Call `get_task_trace(task_id)` -> verify agent `roles`, `assignments`, `messages`, `tool_executions`, `review_decisions`, and `claims_and_evidence_lineage`.
7. **Baseline comparison:** Call `run_single_agent(prompt=..., reference_documents=[...])`, poll `get_task_status`, and compare `get_task_result`.
8. **Cancellation:** Call `run_task(...)` and immediately call `cancel_task(task_id)` -> verify `"cancelled": true` and `get_task_status(task_id)` reports `"cancelled"`.
