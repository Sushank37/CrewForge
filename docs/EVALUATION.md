# CrewForge — Evaluation Report

This document records all CrewForge evaluation experiments, outcomes, and failure analyses without omission or selective reporting.

---

## 1. Historical V1 Held-Out Benchmark (Earlier Engine `81db63a`)

- **Scope:** 25 internally authored cases (`gpt-6-luna`, global `max_output_tokens=2560`).
- **Outcome:** All 25 baseline runs completed; all 25 network runs experienced structured-output truncation (`24/33` specialist calls failed, `29/95` total calls failed with JSON EOF validation errors, `23` partial, `2` failed, `0` completed; `19/25` had zero internal requirement coverage).
- **Root cause:** Reasoning-token starvation under the `2560` output-token ceiling and lack of compact specialist output bounds. Preserved separately as historical baseline evidence; V2 uses a different 10-case prompt set and engine (`612f64e`) and is not a controlled before/after comparison against V1.

---

## 2. Phase 5E Development Canaries (Engine `612f64e`, `gpt-6-luna`)

Following Phase 5E hardening (compact `SpecialistOutput`, `max_output_tokens=8192` ceiling, primary `doc_1..doc_N` provenance preservation, bounded truncation recovery, synthesis preservation, post-gate telemetry), both development canaries completed with **zero unrecovered truncations**:

| Canary | Run ID | Status | Calls | Cost (USD) | Supported Claims | Internal Req. Linkage | Honesty Gate |
|---|---|---|---:|---:|---:|---:|---|
| `canary_a` (financial) | `run_08160b7fa1bd` | `completed` | 6 | \$0.0028538 | 2 / 3 | 83.3% (2 answered, 1 partial) | `full` (`passed=True`) |
| `canary_b` (doc-grounded) | `run_808cff6805b4` | `completed` | 7 | \$0.0038272 | 3 / 4 | 80.0% (3 answered, 2 partial) | `full` (`passed=True`) |

These two genuine traces are exported in `demo/traces/phase5e_canary_a.json` and `demo/traces/phase5e_canary_b.json`.

---

## 3. V2 Same-Model Held-Out Benchmark (Engine `612f64e`, `gpt-6-luna`)

- **Scope:** 10 internally authored held-out cases across 5 domains (2 cases per domain), **not** official hackathon questions.
- **Grading methodology:** **ChatGPT-assisted blind semantic grading** (`chatgpt_blind_v2`, `chatgpt_assisted_semantic_rubric`) using the 100-point rubric (`30/25/20/15/10` with `-5` per unsupported material assertion, max `-20`). Reconciled against `blind_key.json` with zero discrepancy.
- **Execution:** 20 runs total (10 baseline, 10 network), 77 model calls, 0 failed calls, 0 truncations.

### Aggregate Results

| Metric | Single-Agent Baseline (`gpt-6-luna`) | CrewForge Network (`gpt-6-luna`) |
|---|---:|---:|
| **Mean score** | **96.50 / 100** | **55.50 / 100** |
| **Median score** | **99.00 / 100** | **63.00 / 100** |
| **Wins / Losses / Ties (Network)** | — | **1 / 9 / 0** |
| **Completed / Partial / Failed** | 10 / 0 / 0 | 6 / 4 / 0 |
| **Unsupported material assertions** | **1** | **6** |
| **Total cost (USD)** | \$0.0098374 | \$0.0330012 |
| **Mean latency (s)** | 14.25 | 40.79 |
| **Median latency (s)** | 14.46 | 40.96 |

### Domain-Level Scores (2 Cases per Domain)

| Domain | Baseline Mean | Network Mean |
|---|---:|---:|
| `financial` | 98.00 (99, 97) | 65.50 (74, 57) |
| `incident_response` | 98.00 (98, 98) | 42.50 (69, 16) |
| `causation` | 100.00 (100, 100) | 61.00 (47, 75) |
| `document_grounded` | 99.50 (100, 99) | 15.00 (10, 20) |
| `strategic_planning` | 87.00 (100, 74) | **93.50** (87, 100) |

- **Selected scheduling highlight (`v2_str_02`):** Network scored **100** vs Baseline **74** (Baseline violated a deadline constraint on task `D`). This is a single-case result in a 2-case domain; **no statistical significance or general model/network superiority is claimed.**

### Failure Patterns & Epistemic Distinction
1. **Avoidable omissions (major driver of V2 losses):** The network frequently computed accurate intermediate values, then withheld solvable final answers when strict provenance/review checks rejected intermediate steps.
2. **Document-grounding contradictions (`v2_doc_01`, `v2_doc_02`):** All 6 of the network's unsupported material assertions occurred in the 2 document-grounded cases (false absence claims against supplied logs/plans or invented policy rules).
3. **Justified uncertainty vs. safe withholding vs. avoidable omissions:**
   - *Justified uncertainty:* Flagging genuinely missing external facts (e.g., unlogged incident closure date in `canary_b`, missing staffing/SLA cutoffs in `canary_a`).
   - *Safe withholding:* Blocking release of an ungrounded computation rather than fabricating provenance (e.g., `canary_b` monthly uptime % when `43200` was absent from the text).
   - *Avoidable omission:* Failing to ground solvable prompt quantities and withholding directly computable conclusions (e.g., V2 financial/causation losses and Phase 7 rehearsal).

---

## 4. David vs. Goliath (`ollama/qwen3:4b` Network vs. `openai/gpt-6.2-astra` Baseline)

- **Status:** **Not executed — local readiness blocker.**
- **Detail:** Before spending any Astra budget, local `qwen3:4b` (`ollama_think=False`, `ollama_num_ctx=4096`) was tested on `canary_a` (`run_29f576f99bde`). Both initial planning and `plan_repair` generated 4 tasks against `max_tasks=3` (`237.93 s` for 2 calls) and failed before any specialist ran. Zero Astra calls were made (`$0.00` spent).

---

## 5. Phase 7 Live MCP Rehearsal (`run_0f816b1133c1`)

- **Prompt:** Fulfilment capacity (`1,050` orders in `"five hours"`, Line A `120/hr` unavailable first hour, Line B `90/hr` throughout, outsourcing `$2/order`).
- **Outcome:** `status = partial`, `5` model calls, `8,524` in / `4,456` out tokens, `39.04 s`, `$0.0030804`, `honesty_gate_passed = false`.
- **Analysis:** All three numerical calculations were **internally correct** (`930` orders capacity, `120` orders shortfall, `$240` outsourcing cost; `matches_proposed=True` across all 6 deterministic tool calls), but were **withheld** because `"five hours"` (spelled-out word) and derived `a_hours = 4` (`5 - 1`) lacked verbatim digit provenance (`inputs_grounded=False`), causing `reviewer_numbers` to reject `clm_e3f3bfde59be@v2` and `reviewer_consistency` to mark downstream claims `stale`.
