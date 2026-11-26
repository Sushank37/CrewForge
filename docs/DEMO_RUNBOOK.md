# CrewForge — 3-Minute Judge Demo Runbook

## Pre-Flight (30 seconds before demo)
1. Confirm `.env` has `CREWFORGE_ALLOW_PAID_RUNS=false` for safety (enable paid execution only in the temporary MCP server shell if running live cloud inference).
2. Start Mission Control against the dedicated demo database:
   ```bash
   uv run crewforge-dashboard --db-path .crewforge/demo_live.db --host 127.0.0.1 --port 8765
   ```
3. Open `http://127.0.0.1:8765` at desktop width (`1440x900`).

---

## Minute 1: Historical Replay of a Genuine Phase 5E Run
1. On the **Historical Replay** tab (`phase5e_canary_a.json`, `run_08160b7fa1bd`, `gpt-6-luna`), click **Restart** (`0 / 38 events`), then **Step** or **Play**:
   - Before `plan_created`, only the Lead node is visible and the Honesty Gate shows **PENDING REPLAY**.
   - Scrub to full (`38 / 38 events`) to reveal the full **Dynamic Agent Topology** (Lead → Specialists → Reviewers → Honesty Gate).
2. Click a specialist node in the SVG topology to open its **Agent Activity** drawer (assigned tasks + peer messages).

## Minute 2: Verification Inspector, Arithmetic & Honesty Gate
1. In the **Claim Ledger**, click a revised claim (e.g., `clm_0bf2746b2eb9@v2`, `SUPPORTED`) to show:
   - Version history (`v1:rejected → v2:supported`)
   - Supporting evidence (`source_ref`, `source_type`, `verified_in_input`)
   - Deterministic arithmetic tool execution (`expression`, `variables`, `computed_result`, `inputs_grounded`, `matches_proposed`)
   - Independent reviewer verdict & rationale
2. Point out the **Honesty Gate & Final Release** panel: `release_mode`, `gate_passed`, `execution_completed`, `content_released_safely`, and per-requirement linkage (`verification_summary.requirement_statuses`).

## Minute 3: Live MCP Run & Honest Benchmark V2
1. *(If live MCP server is running)* Switch to **Live Monitor** (`Follow newest run` checked), invoke `solve(prompt)` from your MCP client, watch live events stream in, and verify `trace(run_id)` matches the dashboard.
2. Switch to **Benchmark V2** to present the honest evaluation results:
   - **Same-model V2 (`gpt-6-luna`, ChatGPT-assisted blind semantic grading):** Baseline `96.50/100` vs CrewForge `55.50/100` (`1/10` network wins; `strategic_planning` domain `93.50` vs `87.00`; `v2_str_02` `100` vs `74`).
   - **David vs Goliath (`qwen3:4b` vs `gpt-6.2-astra`):** `Not executed — local readiness blocker.`

---

## Offline / Network-Failure Fallback
If API keys, OpenAI network connectivity, or local Ollama are unavailable during the live demo:
1. Stay on **Historical Replay** in Mission Control (`http://127.0.0.1:8765`).
2. Use the genuine exported Phase 5E traces:
   - `phase5e_canary_a.json` (financial capacity & backlog, showing claim revision `v1:rejected → v2:supported` and deterministic arithmetic grounding)
   - `phase5e_canary_b.json` (document-grounded SLA credit policy, showing `doc_1` provenance grounding and safe withholding of ungrounded uptime computation)
3. Clearly state to judges that the replay is a **historical Phase 5E trace** and not live inference.
