# Engine Changes — V2 Freeze

**Previous freeze:** `ce748dc2f82585851204fb13cd2bbf04962655d3`
**Reason for reopening:** Justified reliability and scalability fixes identified during real-user testing.

## Changed Files (5 files, +141 −9 lines)

### 1. `src/crewforge/tools/arithmetic.py` (+126 −1)

**Root cause addressed:** The model generated LaTeX-syntax expressions (e.g., `sum_{k=6}^{10} [C(10,k) * (0.5)^10]`) that the Python `ast.parse()` evaluator rejected, causing computation claims to fail and the reviewer to reject them.

**Changes:**
- Added `import math` for combinatorics support.
- Added `_to_bounded_int()` helper for safe integer conversion with range validation.
- Added allowlisted combinatorics functions:
  - `comb(n, k)` / `C(n, k)` — binomial coefficient
  - `perm(n, k)` / `P(n, k)` — permutation
  - `factorial(n)` — factorial with n ≤ 170 safety bound
  - `binomial_pmf(n, k, p)` — binomial probability mass function $P(X = k)$
  - `binomial_cdf(n, k, p)` — binomial cumulative distribution $P(X \le k)$
  - `binomial_sf(n, k, p)` — inclusive binomial survival function $P(X \ge k)$ (includes threshold $k$)
- Added bounded generator-expression support: `sum(expr for i in range(a, b))` with ≤ 200 iteration safety limit.
- Added cumulative computation safety limit (`_EvalBudget`, `_MAX_CUMULATIVE_OPS = 2500`) across all AST node evaluations, generator iterations, and binomial summation steps.
- All new functions have input validation, overflow protection, and magnitude guards.

### 2. `src/crewforge/prompts/worker.py`

**Changes:**
- Updated specialist prompt to explicitly list Python operators (`+`, `-`, `*`, `/`, `**`) and all new allowed functions, explicitly noting `binomial_cdf(n,k,p)` ($P(X \le k)$) and `binomial_sf(n,k,p)` (inclusive $P(X \ge k)$).
- Added explicit warning against LaTeX `^` or `sum_{...}` syntax — models must use Python syntax.

### 3. `src/crewforge/core/honesty_gate.py` & `src/crewforge/core/orchestrator.py`

**Root cause addressed:** The HonestyGate incorrectly labelled runs as "Partial Verified Release" when zero material claims were actually released, misleading users about verification status.

**Change:**
- `release_mode` in `HonestyGate` is now `"partial"` **only** when `has_usable_output` is True (verified content was actually released). Incomplete runs with zero verified output receive `release_mode = "blocked"`, while `orchestrator.py` independently tracks execution status (`RunStatus.PARTIAL` vs `FAILED`).

### 4. `src/crewforge/prompts/planner.py`

**Changes:**
- Added dynamic team sizing instruction: "Size the team dynamically to match actual task complexity."
- Added anti-inflation guidance: "Never recruit unnecessary agents merely for visual effect."
- Removed the rigid "aim for 1-3" suggestion that constrained legitimate multi-domain tasks.

### 5. `src/crewforge/config.py`

**Changes:**
- `max_workers` and `max_tasks` default to `None` (adaptive capacity within configurable `max_model_calls_per_run` and profile resource limits, up to `le=32` / `le=64` when explicitly set).

## Regression Coverage

New test file: `tests/test_reasoning_scalability.py` (4 tests):
1. `test_deterministic_combinatorics_and_coin_toss_probability` — verifies pure arithmetic for the coin-toss case.
2. `test_coin_toss_probability_network_run_verified_end_to_end` — full mocked NetworkRunner end-to-end.
3. `test_honesty_gate_zero_released_claims_is_blocked_not_partial` — validates the corrected gate behaviour.
4. `test_scalable_eight_specialist_team_and_incremental_events` — 8-specialist team with incremental event delivery.

**All 91 backend tests pass.** The previously failing `test_run_deadline_and_budget_exhaustion_produce_partial_results` now passes correctly.

## Verification Status

| Suite | Result |
|-------|--------|
| `ruff check src tests` | All checks passed |
| `pytest -q` | 102 passed |
| `npm test -- --run` | 8 passed (2 test files) |
| `npm run build` | ✓ 43 modules, production build succeeded |

## V2.1 Authorized Targeted Correctness Fixes (Post-`a816d76`)

1. **`src/crewforge/tools/arithmetic.py` (`build_provenance_catalog`):**
   - Fixed exclusion of `doc_*` `source_ref` entries for primary ingested/retrieved documents (`created_by in {"lead", "baseline", "operator"}`), so MCP-supplied documents (`doc_1`, `doc_2`, ...) resolve deterministically for arithmetic provenance.
   - Preserved alias collision protection (specialist sub-excerpts cannot overwrite `doc_*` aliases), unverified-evidence rejection, fabricated/cross-run `run_id` rejection, and stale claim version rejection.
2. **`src/crewforge/core/repair.py` (`TargetedRepairEngine` & `_build_revision_prompt`):**
   - Constructed the run-scoped `ProvenanceCatalog` in `resolve_disputes()` and included trusted provenance references in targeted repair prompts.

## V2.2 Repair Routing, Atomic Coverage & Evidence Verification Fixes (Post-`ebbf979`)

1. **`src/crewforge/core/reviewer.py`:**
   - Routed rejected material claims with actionable `needs_revision` verdicts into `outcome.blocking_claims` (and downstream invalidation) even when `blocking_severity == "minor"`, while preserving `UNCERTAINTY` / `UNRESOLVED` handling and budget/deadline guards.
2. **`src/crewforge/prompts/worker.py`, `src/crewforge/prompts/planner.py`, `src/crewforge/schemas.py`:**
   - Removed the artificial `1 to 3` claim/deliverable restriction; sized claims dynamically to complexity and required separate atomic claims + deterministic `ToolExecutionRecord`s for each independently requested numerical result.
3. **`src/crewforge/core/evidence_verifier.py` & `src/crewforge/core/ledger.py`:**
   - Verified `external_retrieved` excerpts strictly against their actual referenced document (`retrieved_sources_by_ref`) rather than concatenated unrelated documents, blocking cross-document misattribution.
   - Restricted the negation-prefix check in `_preserves_numbers_and_negation()` to the same sentence/clause so unrelated prior sentences or documents do not cause false negation rejections, while still rejecting omitted same-sentence negation, altered numbers, and noncontiguous/fabricated quotations.

## V2.3 Claim Persistence, Numeric Provenance Context, Dependency-Aware Repair & Evidence Continuity (Post-53bf298)

Targeted domain-independent reliability fixes addressing production task `tsk_pub_ea8ae13f0e1904cc760d6147f295a2fc`:

1. **Eliminated Silent Claim Truncation** (`src/crewforge/core/ledger.py`, `src/crewforge/core/baseline.py`, `src/crewforge/core/worker.py`, `src/crewforge/core/honesty_gate.py`)
   - Removed `claim_proposals[:4]` in `ClaimLedger.record_specialist_outputs()` and `parsed.computations[:4]` in `baseline.py` so all proposed claims and computations within run/call budgets are persisted and reviewed.
   - Removed the fixed `"1 to 2 atomic claims"` instruction in `SpecialistWorker.execute_task()` truncation recovery (`compact_recovery=True`) so required claims and calculations are preserved or explicitly flagged as `is_unresolved` / `missing_information`.
   - Updated `HonestyGate.evaluate()` to explicitly surface omitted requirements in `missing_information` and `Unresolved / Withheld Items` when partial execution or incomplete tasks occur due to resource limits.

2. **Numeric Provenance Context Extraction** (`src/crewforge/tools/arithmetic.py`)
   - Replaced the narrow 45/20-character window and naive `.` / `\n` / `,` splitting in `_extract_numeric_occurrences()` with span-bounded 65-character windows that split only on sentence/paragraph boundaries (`(?<!\d)\.(?!\d)`, `;`, `\n\s*\n`) and inter-number commas, preserving line-break key-value labels, thousands separators, and decimal punctuation while still rejecting swapped quantities, wrong values, fabricated refs, unit conflicts, and stale claim versions.

3. **Dependency-Aware Repair & Fresh Ledger Context** (`src/crewforge/core/repair.py`, `src/crewforge/core/ledger.py`)
   - Refreshed the run trace (`claims`, `evidence`, `tool_executions`) and `ProvenanceCatalog` before each specialist repair batch in `TargetedRepairEngine.resolve_disputes()`.
   - Ordered repair waves topologically so upstream blocking claims are repaired and independently re-reviewed before dependent claims are repaired or revalidated against current supported premise versions.
   - Updated `ClaimLedger.apply_targeted_revision()` to refresh `premise_refs` to latest claim versions rather than preserving superseded premise versions.

4. **Evidence Continuity for Reviewers & Repair Agents** (`src/crewforge/core/reviewer.py`, `src/crewforge/core/repair.py`)
   - Exposed resolved arithmetic input bindings (`variables`, `input_provenance_refs`, and resolved source excerpts) in both `_build_review_user_prompt()` and `_build_revision_prompt()`.
   - Removed the 180-character excerpt truncation and 25-line cap in `_build_revision_prompt()` so repair agents can access full verified source passages without fabricating quotations, and instructed repair agents to preserve existing valid `ToolExecution` records (`computation: null`) during citation-only repairs.
