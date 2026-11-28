# Solution execution and recovery audit

The comparison point is the local V4 candidate `0068a281b127ceeff848e6f5448a73598b8a5f9a`. The historical runs used real model outputs. The changes below were evaluated offline.

## What the historical evidence establishes

| Evidence | Observation | Interpretation |
| --- | --- | --- |
| V1 same-model results, original and recovered | All 20 original network runs failed at the harness call to the missing `NetworkRunner.solve` method. The recovered set contains 18 partial and two failed runs. The Phase 5B.3 diagnosis across 25 runs records 29 failed calls out of 95 attempts, frequently during structured worker output validation. | The original comparison measures a harness failure. The recovered traces also identify lost work before review. Independent semantic grades are unavailable in the V1 same-model report. |
| V2 reconciled blind scores | Network mean 55.5 versus 96.5 for the baseline. Network won the scheduling case 100–74. Its answer included a feasible schedule, an outage infeasibility argument and a revised schedule with a lower-bound argument. It had one scheduling specialist, independent review and three claim revisions. | The network architecture can release a stronger solution. This case alone cannot attribute that gain to peer interaction, decomposition or review. |
| V3 public MCP traces and metrics | Three of five network runs finished partial. All five reported unresolved requirements. Four traces record dependency invalidation, including six invalidation events in finance. The project-reported blind means were 65 for network and 97.4 for baseline. | Execution status alone overstates how much of the user's problem was answered. |
| V4 controlled local finance trace | Both specialists' initial summaries contained the correct scenario totals and choices. The planner bundled several quantities and decisions into single checklist items. The run made 44 arithmetic executions, of which 14 recorded grounded inputs. It used all 16 calls and 86,122 tokens. Five of nine deliverables received released endorsements. The Gate rejected rewritten synthesis text. | Correct reasoning existed before the partial release. Grounding failures, lost operand identity, regeneration and final rendering each contributed. The trace does not establish a quality score for the modified engine. |

The V4 call breakdown was one planner, two workers, four review batches, six specialist revisions, two requirement recoveries and one synthesis. During recovery a correct baseline shipment subtotal of $21,100 was restated as $26,600 even though the recalculated formula still yielded $21,100. Rejecting that replacement was correct.

The source records are under `.crewforge/evaluation/heldout_v1`, `.crewforge/diagnostics/phase5b3`, `.crewforge/evaluation/heldout_v2/same_model`, `.crewforge/v3_public_mcp_eval` and `.crewforge/v4_live_diagnostic/attempt`. The replay takes only a full trace. Test fixtures use independent inputs.

## Pipeline corrections

| Stage | Failure and correction |
| --- | --- |
| Planning and specialist work | The planner could distribute calculations yet leave ownership of the final decision implicit. Planning now asks for a decision owner and its upstream evidence. Specialists write a concise conclusion with its decisive grounds and put those conclusions into reviewable claims. Simple assignments can still use one specialist. |
| Peer challenge | Follow-up read the original plan's empty findings instead of the completed task's findings. It now receives the completed work, the content and version of cited peer claims, and available operand sources. Peers can challenge a concrete assumption before independent review. |
| Source and calculation binding | A partitive sentence such as “Of these” could inherit a counted noun from the preceding sentence while the verifier interpreted that noun as a competing entity. The validator uses a narrowly scoped antecedent. Named, exactly quoted unconditional exclusions can ground zero. An inline percentage complement requires a uniquely bound fraction from a source percentage. Missing derived amounts must still be calculated. Arithmetic and quotations share document aliases, and numbered prompt citations bind to their own clause. |
| Shared knowledge and review | A calculated value previously appeared in the catalog without its statement. A reviewer could see the status of a premise without its meaning. The catalog now pairs the versioned claim, statement, review status, result and tool ID. Reviewers see premise text, source context and relevant previously reviewed answers, and finalize verdicts in dependency order. Unsupported inputs and superseded versions stay ineligible for release. |
| Recovery | Failed response-local intermediates used to disappear from the dependency graph. Their identities now survive failure and can be rebound after correction. The previous root-version cutoff also skipped some grandchildren with higher version numbers. A single bounded lead pass now addresses the remaining checklist across completed assignments, receives specific objections and preserves the original local reference positions. Unrelated proposals are excluded. Identical reviewed calculations can be reused when their operand identities still match. Every new claim returns to independent review. |
| Capacity and release | Repairs can only address their assigned versions. An entirely missing deliverable reserves room for reconciliation and its review. Synthesis selects current reviewed claims and gets their coverage map. Multiple reviewed comparisons or recommendations for the same decision go to the lead for selection. The Gate renders recommendations before workings and can include the verified arithmetic behind a released conclusion. |

The main architectural hypothesis is that coordinated reconciliation and meaningful peer context improve real-model decisions under the call cap. Source matching, reference recovery, call counts and safe rendering have deterministic or scripted tests. Prompt adherence, argument quality and the benefit of peer challenge still need a blinded real-model comparison.

## Offline measurements

Using the original indexed prompt clauses, replaying the 156 prompt-bound operands recorded across all 44 V4 arithmetic executions produced 141 unchanged matches, 11 corrected role conflicts and two newly grounded explicit zero rates. The two remaining failures cite a scenario volume as evidence for a derived tier quantity. This replay measures source binding only. It neither reruns a model nor predicts final benchmark accuracy.

The paired rental comparison used identical inputs and scripted provider behavior on the V4 candidate and the revised production runner. The provider deliberately volunteered the same previously verified rental total during each recovery. Both versions released the four correct deliverables with independent approvals and passed the Gate.

| Paired measurement | V4 candidate | Revised |
| --- | ---: | ---: |
| Covered deliverables | 4/4 | 4/4 |
| Missing comparisons recovered | 2/2 | 2/2 |
| Model calls | 7 | 6 |
| Requirement recovery calls | 2 | 1 |
| Arithmetic executions | 5 | 3 |
| Redundant subtotal executions | 2 | 0 |
| Estimated input tokens, including schemas | 19,033 | 15,736 |

The six independent domain fixtures all ran through `NetworkRunner`, SQLite, the reviewer and the Honesty Gate. The two-specialist numerical fixture released four independently endorsed deliverables in six calls and two arithmetic executions. The document interpretation, causal analysis, constrained optimization, logic and ambiguity fixtures each released the expected decision or qualified conclusion in three calls. A separate release-selection fixture exercised model synthesis in four calls. Another recovery test repaired one bad source with one specialist revision, followed by independent reviews of the corrected subtotal and automatically rebound total. A proposed out-of-scope rewrite never entered the ledger.

The fake provider reports no token usage. The estimates above use CrewForge's reservation estimator on the actual prompts and schemas. Semantic answers and reviewer responses are scripted. These tests establish propagation of the expected solutions through the real engine. Estimating a win rate requires independent real-model outcomes.

## Reproduction

From the repository root, the following commands replay the archived operands and run the independent fixtures offline.

```sh
.venv/bin/python scripts/replay_prompt_operands.py .crewforge/v4_live_diagnostic/attempt/full_trace.json
.venv/bin/python -m pytest -q tests/test_solution_reconciliation.py tests/test_v3_arithmetic_provenance.py -o junit_family=legacy --junitxml=/tmp/crewforge-offline-metrics.xml
```

## Remaining limits

Checklist atomicity and formula applicability still require semantic judgment. A bare zero, an unexplained derived operand or an unknown handle cannot establish empirical evidence. Exact reuse deliberately excludes ambiguous candidates and changed inputs. Algebraically equivalent rewrites can still incur a fresh calculation. A binding or deadline failure can still yield a qualified partial answer under the sixteen-call cap. The strongest unresolved question is whether the revised network improves blind answer quality over a fairly matched single agent.
