# Blinded benchmark

| Historical comparison | Cases | Single agent | Network |
| --- | ---: | ---: | ---: |
| V2 | 10 | 96.5 | 55.5 |
| V3 | 5 | 97.4 | 65.0 |
| V5.1, first fresh case | 1 | 99 | 65 |

On the V5.1 case, both sides used OpenAI gpt-6-luna at low effort. The single agent used 3 calls, 8,255 tokens, 24.62 seconds and $0.0020615. The network used 8 calls, 40,011 tokens, 94.66 seconds and $0.0096883. The network released 8 of 16 deliverables. The release candidate replayed the exposed case offline with scripted model responses and released all 16 in three calls. The replay tests the implementation and does not measure model performance.

Use the next frozen case, `v51_fresh_02`, after the owner approves the allocation. The final release must serve both modes with the same model, effort, input, documents, shared tools and run limits. Keep identities and traces out of the grader packet. Score correctness (30), coverage (25), evidence (20), consistency (15) and usefulness/uncertainty (10). Deduct five points for each distinct unsupported material assertion, capped at twenty. Retain partial and failed outcomes. Compare scores, model calls, tokens, time and cost after identities are revealed. Use the deployed run as the real-assistant acceptance test where it meets this protocol.
