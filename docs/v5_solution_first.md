# V5 solution-first review

The comparison point is `b938fbaa85c2759ee4cff38b8b4e40efd54c33b5` on `fix/v3-verification-recovery`. The two replays below use the same local optimized diagnostic export. The source full-trace SHA-256 is `a83e7e1f0cfc7e0356956cea872363c0483132f8b5252f9cc0a68b91bc69287b`. The first keeps the arithmetic requests fixed. The second exercises recovery and release with scripted messages and fresh runtime IDs.

## Evidence from the original run

The live trace contains 23 successful arithmetic executions with 23 matching proposed results. The previous engine grounded one. Five material roots ended rejected. Twelve dependent material claims and one recommendation were stale. One material claim remained supported. The trace recorded nine calls, 44,740 provider tokens, zero requirement coverage and a reconciled charge of $0.0112832.

Re-executing all 23 requests on the parent reproduced the one grounded record. At the primary sources there were 34 successful bindings, five name-based conflicts and five missing-value failures. The volume quotations contain the cited quantities. The variable-name matcher also inspected other passages containing words from `total_shipments` and rejected the actual selection. Four missing-value failures came from `birch_discount` selecting the complete “No discount.” clause. The old rule required every word in the variable name to belong to its exclusion vocabulary. The fifth missing value was 8,000 additional shipments. Its source says 18,000 total, and its original expression treated 8,000 as an input. A later archived revision explicitly subtracts the 10,000 tier limit.

Downstream failures include rejected temporary aliases, unbound literals whose matching bindings had failed, and invalid premise flags. Two distinct problems become visible once earlier operands bind. Some proposed comparison operands specify USD for results typed USD/month. Two quote strings shorten “Birch’s total monthly cost” to “Birch’s total,” which is absent from the cited statement.

The lead call completed and wrote 15 evidence rows, zero claims and zero tool records. Its raw response is unavailable. The old recovery filter accepted exact outstanding deliverable IDs and their prerequisites, while the specialist system instructions also suggested parent requirement IDs. The trace cannot establish whether the model omitted claims, proposed claims that were filtered, or proposed unusable references. V5 records the parsed response fields, proposed counts, filtered and reused positions, persisted claim IDs and candidate selection counts. Unattached recovery excerpts consume no ledger rows.

## Verification and collaboration

* The arithmetic tool binds primary quantities to exact occurrences, units and conversions. A derived number binds to a particular calculation or versioned claim. An optional derived quotation must occur at that source. A complete explicit exclusion can supply a provisional zero. An unspecified number, missing source, ambiguous occurrence or conflicting unit still fails. The binding record says whether it is invalid or awaiting semantic review.
* The challenger sees the original question, specialist solutions, primary document context, operand selections, units, premises and formulas. Its independent verdict adjudicates entity, scenario, formula applicability and coverage. A reviewer can send a team-level objection to the lead. A mathematically valid binomial calculation still needs a justified independence premise.
* A metadata repair creates a new claim and calculation version. A typed source can supply its original denominator and durable result reference. The engine keeps the proposed value and result as checks. Changed values, incompatible units and unresolved identities require separate correction. The old record and its failure remain available.
* Recovery stages the affected dependency graph in topological order. The reviewer can challenge a provisional chain in one batch. Finalization walks the graph again, requires supported current sources and recomputes the bound formula before promotion. Shared ancestors are valid, cycles are blocked, and a missing or adverse verdict prevents release.
* The lead can select ordered, version-pinned reviewed findings to propose a joint response to a compound deliverable. The application assembles the exact statements. Independent review must endorse the combined answer's completeness. The gate checks the premises and displays overlapping selected findings once. Missing reasoning still goes through the ordinary claim and review path.
* The lead recovery schema has three top-level fields for selections, new claims and evidence. Specialists reference earlier calculations by application-assigned position. Historical proposals with custom aliases remain readable. Both branches use the existing public MCP and dashboard contracts, storage schema and baseline configuration.

## Offline acceptance

Run from the repository root with the local diagnostic export present.

```sh
.venv/bin/python scripts/replay_calculation_graph.py .crewforge/optimized_live_diagnostic/attempt/full_trace.json
.venv/bin/python -m pytest -q tests/test_v5_archive_replay.py tests/test_v5_solution_contract.py
```

| Observation | Result |
| --- | --- |
| V5, original inputs and recorded source selections | 23/23 results matched, 15/23 requests grounded. Eight remained invalid as submitted. |
| Reconstructed solution with new scripted independent reviews | All 18 current financial calculations revalidated. All 19 original claims, including the recommendation, recovered with their 21 original dependency edges pinned to current versions. |
| Reconciliation and release | Seven reviewed joint answers completed the compound items. All nine deliverables and four parent requirements were covered. The gate released the base choice, savings, both higher-volume totals and both ceiling decisions. |
| Execution under the 16-call limit | Ten scripted calls including one specialist revision and four review batches. There were 26 persisted arithmetic executions including failed and superseded attempts. Provider expenditure was zero. |

For the fixed-input replay, the export supplies expressions, variables, refs, units, quotes, proposed results, source text and resolved tool IDs. Temporary aliases and proposal-time claim states are reconstructed from those records. The fresh catalog starts with asserted proposals and obtains its own verdicts in the runner replay. The runner replay reconstructs the accepted plan and first worker proposals from persisted tasks, evidence, versions and edges. It uses the archived corrected tier formula when challenged. The test independently computes expected amounts from the public terms and checks the submitted formulas and citations before issuing scripted verdicts. The new lead selections, reviews and release are counterfactual. Original planner, plan-repair, specialist, reviewer, revision, reconciliation and synthesis provider response bodies and complete request envelopes are unavailable. Live semantic accuracy and provider token usage require a new provider run.

Portable synthetic controls cover variable renames, source reordering, changed entities and quantities, equal-valued wrong-entity or wrong-scenario sources, contextual synonyms, scope of explicit absence, missing inputs, fair trials with unspecified independence, and cyclic and cross-run references. In a separate field-survey problem the estimate used all tagged birds while the field auditor identified recaptures as the population. The scripted challenger supplied the denominator correction. Only the revised ratio and its supporting observation were released in six calls. Earlier changed-premise, fake citation, orphan calculation, stale revision and public MCP tests also exercise the updated contract.

The standalone absence grammar covers quoted “No <term>” clauses. More complex prose can require an explicit sourced interpretation or qualification. Natural-language interpretation, including the meaning of a compound rate or a conditional clause, still depends on the independent reviewer. On a checkout without the local diagnostic export, pytest skips the two archive-dependent tests. The synthetic tests run independently.

## Bounded live validation after owner approval

1. Pin the reviewed V5 SHA and verify a clean checkout. Use a new V5-only directory and exclusive attempt marker. Start an in-process local public MCP service with fresh task and idempotency IDs. Preserve both earlier diagnostic directories.
2. Load only the original public manifest entry for `v3_fin_01` and its supplied documents. Verify the original prompt SHA-256 `ae9050ddab103904885316b64346598b68c28dd0861de5e9a200d593419c0ac1`. Keep the archive, replay logic, expected values and grading materials outside the provider inputs.
3. Submit one network task with OpenAI `gpt-6-luna`, low effort, the previous production profile, at most 16 calls, zero retries and the existing $0.05 run cap. Point isolated run storage at the existing shared budget authority. Reconcile reservations and out-of-ledger spending by identity, including the $0.0192698 V4 and $0.0112832 optimized expenditures exactly once. Check the cumulative limit, global limit, protected reserve and available run envelope before admission. Stop if any check fails.
4. Poll the same task through termination. On interruption recover status or exports from that attempt. Export result, public and full traces, requirement coverage, arithmetic provenance, revision history and usage/latency/cost. Compare per-operand failures, claim and calculation versions, reviewer objections, independent endorsements, decision text, completeness, model calls and reconciled spend.
5. Stop after that diagnostic for owner review, including on failure. A performance claim requires a separately authorized, blinded held-out comparison with identical model, effort, evidence, tools and budget policy for network and single-agent runs.
