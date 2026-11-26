# CrewForge — 3-Minute Judge Demo Runbook

> **Replay vs. Live Execution:** By default, Mission Control opens in **Historical Replay** mode, reconstructing persisted Phase 5E traces (`demo/traces/phase5e_canary_a.json` and `phase5e_canary_b.json`) in recorded event order without making any API calls. Live execution occurs **only** on the **Live Monitor** tab when a separate `crewforge-mcp` or `crewforge` CLI process writes to `--db-path`.

---

## Pre-Flight

1. Launch Mission Control (read-only `mode=ro`):
   ```bash
   uv run crewforge-dashboard --db-path .crewforge/demo_live.db --host 127.0.0.1 --port 8765
   ```
2. Open `http://127.0.0.1:8765` at desktop width (`1440×900`).

---

## Walkthrough Order (3 Minutes)

### 1. Genuine Successful Historical Replay (0:00–0:45)
- On **Historical Replay** (`phase5e_canary_a.json`, `run_08160b7fa1bd`, `gpt-6-luna`), note the **HISTORICAL REPLAY** banner.
- Click **Restart** (`0 / 38 events`), step forward or scrub to `18 / 38` to show that future claims, verdicts, usage, and the Honesty Gate answer remain hidden (`PENDING REPLAY`), then scrub to `38 / 38` (`completed`, `full (safe)`, `83.3%` internal requirement linkage).

### 2. Dynamic Agent Topology (0:45–1:15)
- Inspect the SVG **Dynamic Agent Topology** constructed from the persisted roster (`lead` → specialist agents → `reviewer` → `honesty_gate`).
- Click a specialist node to open the **Agent Activity** drawer showing assigned tasks and peer messages.

### 3. Actual Claim Revision & Arithmetic Verification (1:15–1:50)
- In the **Claim Ledger**, click `clm_0bf2746b2eb9@v2` (`SUPPORTED`) to inspect:
  - Version history (`v1:rejected → v2:supported`)
  - Supporting evidence (`up_1`, `verified=true`)
  - Deterministic arithmetic (`expr: arrivals-(rate*shift_hours) = 800 orders`, `inputs_grounded=true`, `matches_proposed=true`)
  - Independent reviewer verdicts (`needs_revision` on `v1`, `approve` on `v2`)
- Optionally switch to `phase5e_canary_b.json` (`run_808cff6805b4`) to show `doc_1` document-grounded provenance (`95-20 = 75 minutes`, `$1,200` credit) alongside safe withholding of an ungrounded uptime calculation.

### 4. MCP Interoperability (1:50–2:15)
- Show [docs/JUDGE_QUICKSTART.md](JUDGE_QUICKSTART.md) for portable `stdio` and localhost `streamable-http` (`http://127.0.0.1:8000/mcp`) MCP configurations exposing `solve(prompt)`, `baseline(prompt)`, and `trace(run_id)` over the same `.crewforge/demo_live.db`.

### 5. Complete Benchmark Results & Limitations (2:15–2:50)
- Switch to **Benchmark V2**:
  - Present the honest 10-case same-model comparison (`gpt-6-luna`, ChatGPT-assisted blind semantic grading): Baseline **96.50/100** vs CrewForge **55.50/100** (`1/10` network wins; `1` vs `6` unsupported assertions; `strategic_planning` `93.50` vs `87.00`; `v2_str_02` `100` vs `74`).
  - Point out **David vs Goliath**: **`Not executed — local readiness blocker.`**

### 6. Optional Judge-Initiated Live Prompt (2:50–3:00+)
- **Only** if the judge separately authorizes inference with their own API key and budget limits (see [JUDGE_QUICKSTART.md](JUDGE_QUICKSTART.md)): start `crewforge-mcp` with `CREWFORGE_ALLOW_PAID_RUNS=true`, switch Mission Control to **Live Monitor** (`Follow newest run` checked), and invoke `solve(prompt)` from an MCP client.

---

## Historical-Replay Fallback (Network / API Failure)
If API keys or network access are unavailable during the demo, remain on **Historical Replay** using `phase5e_canary_a.json` and `phase5e_canary_b.json`. All telemetry, claims, arithmetic tool results, reviewer verdicts, and Honesty Gate decisions are genuine persisted Phase 5E traces.
