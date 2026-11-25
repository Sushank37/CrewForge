# CrewForge Benchmark Evaluation & Integrity Specification

## 1. Evaluation Methodology & Integrity Guarantees

CrewForge is evaluated under a reproducible, double-blind protocol designed to test whether CrewForge's dynamic multi-agent network produces more correct, complete, and evidence-grounded answers than a competitive single agent—both within the same model class and in an asymmetric small-model network (`qwen3:4b`) versus frontier baseline (`gpt-6.2-astra`) contest.

### Integrity Principles
1. **Zero Contamination:** Engineering and diagnostics use development pilot cases only. Private held-out manifests and grading keys remain external and are never loaded during answer generation.
2. **Separation of Answer Generation & Grading:** `crewforge eval run` generates answers and blinded packets **without** requiring or loading `EvaluationRubric`s or private grading keys.
3. **Deterministic Diagnostics vs. Semantic Grading:** Objective checks report exact-value diagnostics, unit/clause associations, and contradiction candidates, but **never** fabricate a 100-point semantic grade. Ungraded runs remain `pending independent grading` until validated human grades are imported.
4. **Private Random Blinding:** Each experiment generates a cryptographically random 256-bit seed (`secrets.token_hex(32)`), persisted solely in `blind_key.json` and excluded from `evaluator_packet.json` / `evaluator_packet.md`.
5. **Immutable Experiment Fingerprinting:** Every cached run is isolated under `runs/<experiment_fingerprint>/` (hashing manifest, prompt, reference documents, git commit, provider, model, reasoning effort, tools, baseline mode, ablation variant, and runtime limits). Mismatched cached records are rejected.

---

## 2. Standardized 100-Point Evaluation Rubric

Every evaluated answer is scored against the approved 100-point rubric:

| Dimension | Max Points | Description |
| --- | --- | --- |
| **Factual & Numerical Correctness** | 30 | Exact numerical accuracy matching deterministic calculation and empirical facts. |
| **Requirement Coverage** | 25 | Complete coverage of all prompt sub-questions, constraints, and deliverables. |
| **Evidence & Grounding** | 20 | Assertions accurately grounded in supplied source documents or explicit derivations. |
| **Internal Consistency** | 15 | Deductive validity, sound premise-to-conclusion lineage, absence of contradiction. |
| **Usefulness & Uncertainty Handling** | 10 | Actionable clarity, explicit assumptions, and honest identification of missing information. |
| **TOTAL BASE SCORE** | **100** | |
| **Unsupported Assertion Penalty** | **-0 to -20** | **5 points deducted per distinct unsupported material assertion** (capped at 20). |

$$\text{Final Score} = \max\left(0, \text{Base Score} - \min(20, 5 \times \text{Unsupported Material Assertions})\right)$$

---

## 3. David vs. Goliath Configuration

| Role | System | Provider | Model | Pricing (Input / Output per 1M) | Per-Run Ceiling | Category Cap |
| --- | --- | --- | --- | --- | --- | --- |
| **David** | Multi-Agent Network | `ollama` | `qwen3:4b` | Local ($0.00) | $0.00 | — |
| **Goliath** | Single-Agent Baseline | `openai` | `gpt-6.2-astra` | $8.00 / $40.00 (short-context) | $0.20 | $1.10 |

- All David agents strictly use `qwen3:4b` (`small_model_profile=True`, `concurrency=1`, `max_workers=2`, `max_tasks=3`, `ollama_num_ctx=4096`). No stronger model is permitted inside David.
- Goliath runs exclusively via single-agent `BaselineRunner` with `DeterministicArithmeticTool` and identical supplied reference documents—no multi-agent delegation.
- Actual `returned_model` identities are recorded; if a model cannot be accessed, the run is recorded as `failed`/`unavailable` with zero silent model substitution.

---

## 4. Compute-Budget Comparison Fairness

When running `--mode compute_matched_vs_network`:
- `BaselineRunner` accepts configurable `--baseline-max-calls` and `--baseline-max-output-tokens` allowances.
- Iterative self-critique rounds (up to 3) receive the same permitted reference documents and incorporate newly requested `DeterministicArithmeticTool` results before finalizing.
- Actual calls, tokens, latency, cost, and **unused compute** (`unused_calls`, `unused_output_tokens`) are recorded honestly.
- When actual consumption differs from the network, the comparison is described as **budget-matched** rather than exactly compute-matched.

---

## 5. Genuine Ablation Suite & Experimental Confounds

1. `full`: Production configuration (peer messaging, independent review, targeted repair, Honesty Gate).
2. `no_reviewer`: Disables `ReviewerPool` completely (0 reviewer model calls). **Experimental Confound:** Because claims cannot reach `SUPPORTED` status without independent review, the active Honesty Gate withholds unreviewed factual claims from verified release rather than falsely labeling them as verified.
3. `no_repair`: Retains `ReviewerPool` but disables `TargetedRepairEngine` (0 revision calls). **Experimental Confound:** Rejected claims remain unrevised, reducing released coverage when initial worker claims are flawed.
4. `no_peer_messages`: Disables inter-specialist message reading/dispatch and follow-up rounds (0 peer messages delivered).

---

## 6. Budget Protection & Future Finale Profile

- Canonical `UnifiedBudgetAuthority` (`.crewforge/budget_authority.db`) tracks all historical and benchmark spending ($0.0255164 committed across 55 unique calls).
- The **$4.00 emergency protected reserve** cannot be touched by development or Phase 5 benchmarks (`phase5_same_model` cap $0.75, `david_vs_goliath` cap $1.10).
- **Future Explicitly Authorized Finale Profile:** To use the reserved $4.00 allocation during the live hackathon finale without weakening development protection, an operator must explicitly pass a dedicated finale profile (setting `CREWFORGE_ALLOW_PAID_RUNS=true`, `CREWFORGE_PROTECTED_RESERVE_USD=0.0`, and `CREWFORGE_BUDGET_ALLOCATION_CATEGORY=finale` in an isolated finale environment) after Phase 5 evaluation is complete. This profile is **not** activated in code or `.env`.

---

## 7. Exact Runnable Evaluation Commands

1. **Validate an external manifest (resolving relative docs & SHA-256 hashes):**
   ```bash
   .venv/bin/crewforge eval validate-manifest --manifest /path/to/held_out_manifest.json
   ```
2. **Run same-model benchmark without loading answer rubrics:**
   ```bash
   .venv/bin/crewforge eval run --manifest /path/to/held_out_manifest.json --output-dir benchmarks/results/same_model --mode baseline_vs_network --no-resume
   ```
3. **Resume the same experiment safely:**
   ```bash
   .venv/bin/crewforge eval run --manifest /path/to/held_out_manifest.json --output-dir benchmarks/results/same_model --mode baseline_vs_network --resume
   ```
4. **Execute compute-budget comparison subset:**
   ```bash
   .venv/bin/crewforge eval run --manifest /path/to/held_out_manifest.json --output-dir benchmarks/results/budget_matched --mode compute_matched_vs_network --baseline-max-calls 6 --baseline-max-output-tokens 4096
   ```
5. **Execute asymmetric David (`ollama/qwen3:4b`) vs. Goliath (`openai/gpt-6.2-astra`) comparison:**
   ```bash
   .venv/bin/crewforge eval run --manifest /path/to/held_out_manifest.json --output-dir benchmarks/results/david_vs_goliath --david-vs-goliath
   ```
6. **Execute predetermined ablation subset (e.g., `no_reviewer`, `no_repair`, `no_peer_messages`):**
   ```bash
   .venv/bin/crewforge eval run --manifest /path/to/held_out_manifest.json --output-dir benchmarks/results/ablation_no_reviewer --mode ablation --ablation-variant no_reviewer
   ```
7. **Export blinded evaluator packets:**
   Automatically written during `eval run` to `<output-dir>/evaluator_packet.json` and `<output-dir>/evaluator_packet.md` (with `blind_key.json` persisted separately).
8. **Import completed human grades, unblind, and generate final report:**
   ```bash
   .venv/bin/crewforge eval grade --results benchmarks/results/same_model/experiment_results.json --grades /path/to/human_grades.json --blind-key benchmarks/results/same_model/blind_key.json --rubrics /path/to/private_rubrics.json --output-dir benchmarks/results/same_model/graded
   ```
9. **View unblinded final report:**
   Inspect `benchmarks/results/same_model/graded/final_report.md` and `final_results.csv`.
10. **Export genuine historical demonstration trace (prompt redacted by default):**
    ```bash
    .venv/bin/crewforge export-demo run_17a4f645e9ba --output benchmarks/demo_trace_run_17a4f645e9ba.json
    ```
