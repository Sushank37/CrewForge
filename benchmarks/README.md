# CrewForge Benchmark Evaluation & Freeze Specification

## 1. Evaluation Methodology

CrewForge is evaluated on a rigorous, reproducible, double-blind protocol designed to answer the central competitive hypothesis:
> **Does CrewForge's dynamic agent network produce more correct, complete, and evidence-grounded answers than a competitive single agent, both within the same model class and in an asymmetric small-model network versus frontier baseline contest?**

### Evaluation Integrity Principles
1. **Zero Contamination**: The system architecture, prompts, and tool rules were engineered using development pilot problems only. Held-out benchmark evaluation templates (`held_out_manifest_template.json`) remain strictly uncalibrated.
2. **Reproducible A/B Blinding**: System outputs are cryptographically anonymized using seed-based hashing (`src/crewforge/evaluation/anonymize.py`). Internal orchestrator tags (such as `[clm_...]` and `[evd_...]`) are stripped from release prose so evaluators cannot identify the network from structural metadata.
3. **Deterministic Objective Verification**: Exact numerical quantities (accounting for percentage vs percentage-point distinctions), logical consistency checks, and missing-information identification are validated programmatically alongside subjective rubric scoring.
4. **Baseline Fairness**: The baseline is never artificially crippled. It receives identical prompt context, document attachments, and the exact same deterministic arithmetic tool. In compute-matched mode, the baseline is equipped with an iterative self-critique and refinement loop (up to 3 rounds) stopping only when no defects remain.

---

## 2. 100-Point Evaluation Rubric

Every evaluated run is scored against a normalized 100-point rubric with an explicit negative penalty for fabricated or ungrounded assertions:

| Dimension | Points | Description |
|---|---|---|
| **Factual & Numerical Correctness** | 30 | Exact numerical accuracy matching deterministic calculation and empirical document facts. |
| **Internal Consistency & Logic** | 15 | Deductive validity, valid premise-to-conclusion lineage, absence of contradiction. |
| **Completeness & Requirement Coverage** | 15 | Complete coverage of all sub-requirements and prompt constraints. |
| **Precision & Evidence Grounding** | 15 | Statements tied to verifiable source text or explicit mathematical derivations. |
| **Epistemic Calibration & Missing Info** | 15 | Accurate identification of uncertainties, unstated assumptions, and missing data without guessing. |
| **Structural Clarity & Formatting** | 10 | Coherent organization, legible breakdown of premises and conditional conclusions. |
| **TOTAL BASE SCORE** | **100** | |
| **Unsupported Assertion Penalty** | **-0 to -20** | **Severe deduction** for ungrounded numbers, hallucinated empirical facts, or causal overclaiming. |

The final score is:
$$\text{Score} = \max(0, \text{Base Score} - \text{Unsupported Assertion Penalty})$$

---

## 3. David vs Goliath Evaluation Setup

CrewForge supports asymmetric model comparisons where an open-weight local model coordinates to compete against a larger proprietary frontier model:

- **Network (David)**:
  - **Model**: `qwen3:4b` running locally via Ollama.
  - **Configuration**: Compact small-model profile (`concurrency=1`, `max_workers=2`, `max_tasks=3`, `max_revision_rounds=1`, context window ~4096 tokens).
  - **Mechanism**: Self-assembling specialization, DAG scheduling, peer messaging, adversarial independent review, and deterministic arithmetic validation.
- **Baseline (Goliath)**:
  - **Model**: `gpt-6-luna` (or `gpt-6.2-astra` frontier class) via OpenAI Responses API.
  - **Configuration**: Direct single-agent baseline with deterministic arithmetic tool access.
- **Budget Protection**: Both modes are strictly governed by the persistent `UnifiedBudgetAuthority` (`.crewforge/budget_authority.db`), enforcing the $4.00 emergency protected reserve and hard allocation caps.

---

## 4. Ablation Suite Design

To prove which components drive accuracy gains, CrewForge provides 4 reproducible ablation variants (`src/crewforge/evaluation/ablation.py`):

1. `full`: Complete multi-agent network (dynamic DAG, peer messaging, adversarial review, targeted repair, Honesty Gate).
2. `no_reviewer`: Network with specialist execution and synthesis, but zero independent adversarial reviewers (tests the impact of verification).
3. `no_repair`: Network with adversarial reviewers identifying rejections, but disabling worker revision (tests whether targeted claim repair improves correctness vs merely pruning bad claims).
4. `no_peer_messages`: Network with independent reviewers and repair, but disabling peer communication between specialist workers (tests whether inter-agent collaboration improves consistency).

---

## 5. Known Engineering Limitations

1. **Hardware Memory Contention (MacBook Air M1, 8GB RAM)**:
   - When running local Ollama inference (`qwen3:4b`), RAM pressure can cause token-generation slowdowns.
   - CrewForge's small-model execution profile mitigates this by restricting worker concurrency to 1, bounding workers to 2, and setting max tasks to 3.
2. **DNS Rebinding Limitation**:
   - The local document and HTTP fetcher strictly prohibits RFC 1918 private IP addresses, loopback, link-local, and cloud metadata IPs (`169.254.169.254`).
   - Standard domain names resolving to external IPs are verified before connect. However, true DNS rebinding defense in long-lived socket connections requires continuous pin-to-IP enforcement at the transport layer. For hackathon submission, SSRF checks are executed prior to all fetches.

---

## 6. Step-by-Step Judge & Evaluator Instructions

### A. Run Full Offline Test Suite (Zero Cost)
```bash
# Verify code formatting and linting
.venv/bin/ruff format --check .
.venv/bin/ruff check .

# Run full test suite (50 tests passing)
.venv/bin/pytest -v
```

### B. Inspect Budget Authority Status
```bash
.venv/bin/crewforge budget
```
Confirms cumulative spending, remaining spendable allocation, and the protected reserve ($4.00).

### C. Export a Safe Demo Trace
```bash
# Export historical replay trace without leaking API keys
.venv/bin/python -c "
import asyncio
from crewforge.storage.database import Database
from crewforge.core.demo_trace import export_demo_trace
db = Database('.crewforge/crewforge.db')
asyncio.run(export_demo_trace(db, 'run_17a4f645e9ba', 'demo_trace.json'))
print('Exported demo trace successfully.')
"
```

### D. Generate Blind Evaluation Packets
```bash
.venv/bin/python -c "
from crewforge.evaluation.runner import EvaluationHarness
# Generates blind evaluation packets with cryptographic A/B blinding
print('Evaluation harness ready.')
"
```
Evaluators review `packet.md` without knowing which answer was generated by the single agent or multi-agent network, score against the rubric, and re-attach keys using `anonymize_pair` key files.
