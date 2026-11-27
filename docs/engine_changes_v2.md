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
  - `binomial_pmf(n, k, p)` — binomial probability mass function
  - `binomial_cdf(n, k, p)` — binomial cumulative distribution
  - `binomial_sf(n, k, p)` — binomial survival function (1 − CDF)
- Added bounded generator-expression support: `sum(expr for i in range(a, b))` with ≤ 200 iteration safety limit.
- All new functions have input validation, overflow protection, and magnitude guards.

### 2. `src/crewforge/prompts/worker.py` (+8 −4)

**Changes:**
- Updated specialist prompt to explicitly list Python operators (`+`, `-`, `*`, `/`, `**`) and all new allowed functions.
- Added explicit warning against LaTeX `^` or `sum_{...}` syntax — models must use Python syntax.

### 3. `src/crewforge/core/honesty_gate.py` (+1 −1)

**Root cause addressed:** The HonestyGate incorrectly labelled runs as "Partial Verified Release" when zero material claims were actually released, misleading users about verification status.

**Change:**
- `release_mode` is now `"partial"` only when `has_usable_output` OR `is_partial_run` with unresolved/missing items. Previously, any unresolved or missing items (even without usable output) triggered `"partial"`. Now, without usable output AND without partial-run context, the mode is correctly `"blocked"`.

### 4. `src/crewforge/prompts/planner.py` (+5 −1)

**Changes:**
- Added dynamic team sizing instruction: "Size the team dynamically to match actual task complexity."
- Added anti-inflation guidance: "Never recruit unnecessary agents merely for visual effect."
- Removed the rigid "aim for 1-3" suggestion that constrained legitimate multi-domain tasks.

### 5. `src/crewforge/config.py` (+2 −2)

**Changes:**
- `max_workers`: default 4→6, upper bound 8→32 (supports scalable agent teams).
- `max_tasks`: default 6→8, upper bound 12→64 (supports complex multi-task plans).

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
| `pytest -q` | 91 passed |
| `npm test -- --run` | 8 passed (2 test files) |
| `npm run build` | ✓ 43 modules, production build succeeded |
